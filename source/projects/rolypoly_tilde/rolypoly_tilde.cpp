// 2023 rvirmoors
// adapted from nn~ by Antoine Caillon & Axel Chemla-Romeu-Santos

#define DEBUG true

#ifndef VERSION
#define VERSION "2.0b1"
#endif

// midi stuff
#include "MidiFile.h"
#define MAX_SCORE_LENGTH 100000
#define SCORE_DIM 6 // hit, vel, bpm, tsig, pos_in_bar, TIME_MS
#define IN_DIM 5    // hit, vel, bpm, tsig, pos_in_bar
#define OUT_DIM 14  // 9 hits, bpm, tsig, pos_in_bar, TAU_drum, TAU_guitar
#define TIME_MS 5
#define TAU 12

// nn~
#include "c74_min.h"
#include "torch/torch.h"
#include "../../../nn_tilde/src/backend/backend.h"
#include "../../../nn_tilde/src/frontend/maxmsp/shared/circular_buffer.h"
#include <string>
#include <thread>
#include <vector>
#include <chrono>

using namespace c74::min;
using namespace smf;

unsigned power_ceil(unsigned x) {
  if (x <= 1)
    return 1;
  int power = 2;
  x--;
  while (x >>= 1)
    power <<= 1;
  return power;
}

class rolypoly : public object<rolypoly>, public vector_operator<> {
public:
	MIN_DESCRIPTION {"Expressive Drum Machine: read MIDI file, listen to audio, output drums"};
	MIN_TAGS {"drums, sync, deep learning"};
	MIN_AUTHOR {"Grigore Burloiu // rvirmoors"};
	MIN_RELATED {"nn~, flucoma~"};

	// INLETS OUTLETS
	std::vector<std::unique_ptr<inlet<>>> m_inlets;
	std::vector<std::unique_ptr<outlet<>>> m_outlets;

	rolypoly(const atoms &args = {});
	~rolypoly();

  // MIDI RELATED MEMBERS
  MidiFile midifile;
  c74::min::path m_midi_path;
  double** score;
  long i_toModel; // next timestep to be sent to the model
  long t_score; // next timestep to be played from score
  long score_size;
  int reading_midi;  
  int skip; // used to skip everything but NoteOn events
  bool done_reading;

  double playhead_ms;  // in ms
  long i_fromModel; // next timestep to be read from the model  
  long t_play; // next timestep to be played from play_notes
  std::vector<std::array<double, IN_DIM>> in_notes; // notes to be sent to the model
  std::array<double, IN_DIM> in_onset; // onset to be sent to the model
  at::Tensor modelOut; // result from calling model.forward()
  std::vector<std::array<double, OUT_DIM>> play_notes; // hits to be played
  bool done_playing;
  int lookahead_ms; // in ms
  short timer_mode; // 0 inactive, 1 read, 2 play
  enum TIMER {INACTIVE, READ, PLAY};

  std::vector<std::pair<long, double>> tempo_map;
  int current_tempo_index;
  std::vector<std::pair<long, double>> timesig_map;
  int current_timesig_index;

  void initialiseScore();
  void parseTimeEvents(MidiFile &midifile);
  //void resetInputBuffer();
  bool midiNotesToModel();

  void prepareToPlay();
  void playMidiIntoVector();
  void vectorToModel(std::vector<std::array<double, IN_DIM>> &v);
  void getTauFromModel();
  double computeNextNoteTimeMs();
  void incrementPlayIndexes();
  void processLiveOnsets(audio_bundle input);

	// BACKEND RELATED MEMBERS
	Backend m_model;
	std::string m_method;
	std::vector<std::string> settable_attributes;
	bool has_settable_attribute(std::string attribute);
	c74::min::path m_path;
  int m_in_dim, m_out_dim;

	// BUFFER RELATED MEMBERS
	int m_buffer_size;
	std::unique_ptr<circular_buffer<double, float>[]> m_in_buffer;
	std::unique_ptr<circular_buffer<float, double>[]> m_out_buffer;
	std::vector<std::unique_ptr<float[]>> m_in_model, m_out_model;

	// AUDIO PERFORM
  bool m_use_thread;
	std::unique_ptr<std::thread> m_compute_thread;
	void operator()(audio_bundle input, audio_bundle output);
	void buffered_perform(audio_bundle input, audio_bundle output);
	void perform(audio_bundle input, audio_bundle output);
  void model_perform();
  void wait_a_sec();


  // ONLY FOR DOCUMENTATION
  argument<symbol> path_arg{this, "model path",
                            "Absolute path to the pretrained model."};
  argument<symbol> method_arg{this, "method",
                              "Name of the method to call during synthesis."};
  argument<int> buffer_arg{
      this, "buffer size",
      "Size of the internal buffer (can't be lower than the method's ratio)."};

  // ENABLE / DISABLE ATTRIBUTE
  attribute<bool> enable{this, "enable", true,
                         description{"Enable / disable tensor computation"}};

  // LATENCY (guitar onsets) ATTRIBUTE
  attribute<int> latency{this, "latency", 512,
                         description{"Onset detection latency (samples)"}};

  // BOOT STAMP
  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        cout << "rolypoly~ v" << VERSION << " - 2023 Grigore Burloiu - rvirmoors.github.io" << endl;
        cout << "adapted from nn~ by Antoine Caillon & Axel Chemla-Romeu-Santos" << endl;
        return {};
      }};

  message<> anything {this, "anything", "callback for attributes",
    MIN_FUNCTION {
      symbol attribute_name = args[0];
      if (attribute_name == "get_attributes") {
        for (std::string attr : settable_attributes)
          cout << attr << endl;
        return {};
      } 
      else if (attribute_name == "get_methods") 
      {
        for (std::string method : m_model.get_available_methods()) 
          cout << method << endl;
        return {};
      } 
      else if (attribute_name == "get") 
      {
        if (args.size() < 2) {
          cerr << "get must be given an attribute name" << endl;
          return {};
        }
        attribute_name = args[1];
        if (m_model.has_settable_attribute(attribute_name)) {
          cout << attribute_name << ": " << m_model.get_attribute_as_string(attribute_name) << endl;
        } else {
          cerr << "no attribute " << attribute_name << " found in model" << endl;
        }
        return {};
      }
      else if (attribute_name == "set") 
      {
        if (args.size() < 3) {
          cerr << "set must be given an attribute name and corresponding arguments" << endl;
          return {};
        }
        attribute_name = args[1];
        std::vector<std::string> attribute_args;
        if (has_settable_attribute(attribute_name)) {
          for (int i = 2; i < args.size(); i++) {
            attribute_args.push_back(args[i]);
          }
          try {
            m_model.set_attribute(attribute_name, attribute_args);
          } catch (std::string message) {
            cerr << message << endl;
          }
        } else {
          cerr << "model does not have attribute " << attribute_name << endl;
        }
      }
      else
      {
        cerr << "no corresponding method for " << attribute_name << endl;
      }
      return {};
     }};

  std::string attr;
  std::string attr_value;
  queue<> set_attr { this,
      MIN_FUNCTION {
        // send low-priority messages to the python model
        std::vector<std::string> v;
        v.push_back(attr_value);
        m_model.set_attribute(attr, v);
        return {};
      }
  };

  timer<timer_options::defer_delivery> warmup { this,
    MIN_FUNCTION {
      // warmup the model
      std::chrono::microseconds duration;
      for (int i = 0; i < 7; i++) {
          auto start = std::chrono::high_resolution_clock::now();
          model_perform();
          auto end = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
          if (DEBUG) cout << "step " << i+1 << "/7: " << duration.count() / 1000. << " ms" << endl;
      }
      // if the model is too slow, we need to increase the lookahead
      if (duration.count() / 1000. > lookahead_ms / 4) {
        lookahead_ms = duration.count() / 1000. * 4 + 50;
        if (DEBUG) cout << "increasing lookahead to " << lookahead_ms << " ms" << endl;
      }
      return {};
    }
  };

  timer<timer_options::defer_delivery> m_timer { this, MIN_FUNCTION {
    cout << "t_score and size ============  " << t_score << "    <<<<  " << score_size << endl;
    if (timer_mode == TIMER::READ) {
      //cout << "timer read" << endl;
      read_deferred.set();
    } else if (timer_mode == TIMER::PLAY) {
      //cout << "timer play" << endl;
      perform_threaded.set();
      if (!done_playing) {
        m_timer.delay(lookahead_ms / 2);
      }
    } 
    return {};
  }};

  queue<> read_deferred {this, 
    MIN_FUNCTION {
      //resetInputBuffer();
      reading_midi ++;
      if (DEBUG) cout << reading_midi << " " << done_reading << endl;

      if (reading_midi && !done_reading) {
        // populate score and place it into m_in_buffer
        done_reading = midiNotesToModel();
        for (int c(0); c < IN_DIM; c++) {
          m_in_buffer[c].put(score[c], m_buffer_size);
        }
        if (DEBUG) cout << "input buffers read: " << reading_midi << endl;
      } else {
        // fill m_in_buffer with zeros
        for (int c(0); c < IN_DIM; c++) {
          m_in_buffer[c].reset();
        }
      }
      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < IN_DIM; c++)
        m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);
        // TODO: what if the buffer is full of midi notes but not done_reading? Does reading several buffers work?

      // PROCESS SCORE
      model_perform();

      if (done_reading && reading_midi) {
        if (DEBUG) cout << "done reading" << endl;
        reading_midi = 0;
        attr = "read"; attr_value = "false"; set_attr();
        prepareToPlay();
      } else {
        m_timer.delay(10);
      }
      return {};
    }
  };

  queue<> perform_threaded { this,
    MIN_FUNCTION {
      if (m_compute_thread && m_compute_thread->joinable()) {
        if (DEBUG) cout << "joining - performing " << playhead_ms << endl;
        //if (DEBUG) cout << m_compute_thread->get_id() << endl;
        m_compute_thread->join();
        if (DEBUG) cout << "joined at " << playhead_ms << " ms : " << i_toModel << endl;
      }

      // send midi notes
      // to the circular buffer, to later get the model output
      if (!done_playing)
        playMidiIntoVector();

      // if (m_in_buffer[0].full()) {
        // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
        // for (int c(0); c < IN_DIM; c++)
        //   m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);

        // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
        // for (int c(0); c < OUT_DIM; c++)
        //   m_out_buffer[c].put(m_out_model[c].get(), m_buffer_size);

        //getTauFromModel();
      // }

      if (m_use_thread && !done_playing) {
        //m_compute_thread = std::make_unique<std::thread>(&rolypoly::vectorToModel, this, play_notes);
        //if (DEBUG) cout << "started thread" << endl;
      }
      return {};
    }
  };

  message<> read {this, "read", "Load score",
    MIN_FUNCTION {
      done_reading = false;
      attr = "read"; attr_value = "true"; set_attr();
      timer_mode = TIMER::READ;
      m_timer.delay(0);
      return {};
    }
  };

  message<> start {this, "start", "Start playing",
    MIN_FUNCTION {
      if (!score_size) {
        cerr << "no score loaded, can't play yet!" << endl;
        return {};
      }
      done_playing = false;
      prepareToPlay();
      attr = "play"; attr_value = "true"; set_attr();
      timer_mode = TIMER::PLAY;
      m_timer.delay(0);
      
      return {};
    }
  };

};

void rolypoly::wait_a_sec() {
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

void rolypoly::model_perform() {
  std::vector<float *> in_model, out_model;
  for (int c(0); c < IN_DIM; c++)
    in_model.push_back(m_in_model[c].get());
  for (int c(0); c < OUT_DIM; c++)
    out_model.push_back(m_out_model[c].get());
  m_model.perform(in_model, out_model, m_buffer_size, m_method, 1);
}

void rolypoly::initialiseScore() {\
  // delete score
  if (score_size) {
    for (int c = 0; c < SCORE_DIM; c++) {
        delete[] score[c];
    }
    delete[] score;
  }
  score = new double*[SCORE_DIM];
  for (int c(0); c < SCORE_DIM; c++) {
    score[c] = new double[MAX_SCORE_LENGTH];
  }
}

rolypoly::rolypoly(const atoms &args)
    : m_compute_thread(nullptr), score_size(0),
      m_buffer_size(64), m_method("forward"),
      m_use_thread(true), lookahead_ms(200) {

  m_model = Backend();

  // CHECK ARGUMENTS
  if (!args.size()) {
    return;
  }
  if (args.size() > 0) { // ONE ARGUMENT IS GIVEN
    auto model_path = std::string(args[0]);
    if (model_path.substr(model_path.length() - 3) != ".ts")
      model_path = model_path + ".ts";
    m_path = path(model_path);
  }
  if (args.size() > 1) { // TWO ARGUMENTS ARE GIVEN
    auto midi_path = std::string(args[1]);
    if (midi_path.substr(midi_path.length() - 4) != ".mid")
      midi_path = midi_path + ".mid";
    m_midi_path = path(midi_path);
    cout << "midi path: " << midi_path << endl;
    midifile.read(std::string(m_midi_path));
    if (!midifile.status()) {
      cerr << "Error reading MIDI file " << std::string(m_midi_path) << endl;
    }
    midifile.linkNotePairs();    // first link note-ons to note-offs
    midifile.doTimeAnalysis();   // then create ticks to seconds mapping
    parseTimeEvents(midifile);

    initialiseScore();
    done_playing = false;
    playhead_ms = i_fromModel = i_toModel = 
      t_score = t_play =
      current_tempo_index = current_timesig_index = 
      skip = 0;
  }
  if (args.size() > 2) { // THREE ARGUMENTS ARE GIVEN
    m_buffer_size = int(args[2]);
  }

  // TRY TO LOAD MODEL
  if (m_model.load(std::string(m_path))) {
      cerr << "error during loading" << endl;
      error();
      return;
  }

  // GET MODEL'S METHOD PARAMETERS
  auto params = m_model.get_method_params(m_method);

  // GET MODEL'S SETTABLE ATTRIBUTES
  try {
    settable_attributes = m_model.get_settable_attributes();
  } catch (...) { }

  if (!params.size()) {
    error("method " + m_method + " not found !");
  }

  m_in_dim = params[0];
  m_out_dim = params[2];

  if (m_buffer_size < 16) {
    cerr << "buffer size too small, should be at least 16" << endl;
    m_buffer_size = 16;
  }
  m_buffer_size = power_ceil(m_buffer_size);

  // Calling forward in a thread causes memory leak in windows.
  // See https://github.com/pytorch/pytorch/issues/24237
//#ifdef _WIN32
//  m_use_thread = false;
//#endif

  // CREATE INLET, OUTLETS and BUFFERS
  m_inlets.push_back(std::make_unique<inlet<>>(
    this, "(signal) musician input", "signal"));
  m_in_buffer = std::make_unique<circular_buffer<double, float>[]>(m_in_dim);
  for (int i(0); i < m_in_dim; i++) {
    m_in_buffer[i].initialize(m_buffer_size);
    m_in_model.push_back(std::make_unique<float[]>(m_buffer_size));
  }

  m_out_buffer = std::make_unique<circular_buffer<float, double>[]>(m_out_dim);
  for (int i(0); i < m_out_dim; i++) {
    std::string output_label = "";
    try {
      output_label = m_model.get_model().attr(m_method + "_output_labels").toList().get(i).toStringRef();
    } catch (...) {
      output_label = "(signal) model output " + std::to_string(i);
    }
    m_outlets.push_back(std::make_unique<outlet<>>(
        this, output_label, "signal"));
    m_out_buffer[i].initialize(m_buffer_size);
    m_out_model.push_back(std::make_unique<float[]>(m_buffer_size));
  }
  cout << "Running warmup, please wait (Max will freeze for a few seconds) ..." << endl;
  // "play must be set to true for this to work"
  attr = "play"; attr_value = "false"; set_attr();  
  warmup.delay(500);
}

rolypoly::~rolypoly() {
  if (m_compute_thread && m_compute_thread->joinable())
    m_compute_thread->join();
  // delete score
  if (score_size) {
    for (int c = 0; c < SCORE_DIM; c++) {
        delete[] score[c];
    }
    delete[] score;
  }
}

bool rolypoly::has_settable_attribute(std::string attribute) {
  for (std::string candidate : settable_attributes) {
    if (candidate == attribute)
      return true;
  }
  return false;
}

void fill_with_zero(audio_bundle output) {
  for (int c(0); c < output.channel_count(); c++) {
    auto out = output.samples(c);
    for (int i(0); i < output.frame_count(); i++) {
      out[i] = 0.;
    }
  }
}

void rolypoly::parseTimeEvents(MidiFile &midifile) {
  for (int i=0; i<midifile.getNumEvents(0); i++) {
    int command = midifile[0][i][0] & 0xf0;
    if (midifile[0][i][0] == 0xff && midifile[0][i][1] == 0x51) {
      // tempo change
      int microseconds = (midifile[0][i][3] << 16) |
                          (midifile[0][i][4] << 8) |
                          midifile[0][i][5];
      double bpm = 60.0 / microseconds * 1000000.0;
      if (DEBUG) cout << "Tempo change at tick " << midifile[0][i].tick
           << " to " << bpm << " beats per minute" << endl;
      tempo_map.push_back(std::make_pair(midifile[0][i].tick, bpm));
    }
    if ((midifile[0][i][0] & 0x0f) == 0x09) {
      continue;
    }
    if (command == 0xf0) {
      command = midifile[0][i][0];
    } 
    if (command == 0xff && midifile[0][i][1] == 0x58) {
      // time signature change
      int numerator = midifile[0][i][3];
      int denominator = midifile[0][i][2];
      if (DEBUG) cout << "Time signature change at tick " << midifile[0][i].tick
            << " to " << numerator << "/" << denominator << endl;
      timesig_map.push_back(std::make_pair(midifile[0][i].tick, (double)numerator / denominator));
    }
  }
}

// void rolypoly::resetInputBuffer() {
//   if(!m_in_buffer[0].empty() && !reading_midi && !done_reading) {
//     if (DEBUG) cout << "resettin" << endl;
//     // if the buffer isn't empty, reset it
//     for (int c(0); c < IN_DIM; c++)
//       m_in_buffer[c].reset();
//     done_reading = false;
//     if (DEBUG) cout << "starting to read" << endl;
//   }
// }

bool rolypoly::midiNotesToModel() {
  // populates score
  int channels = SCORE_DIM;
  int startFrom = (reading_midi - 1) * m_buffer_size + skip;
  // if done, then fill with zeros
  if (startFrom >= midifile[1].size()) {
    if (DEBUG) cout << "reached end of midifile" << endl;
    for (int c = 0; c < channels; c++) {
      for (int i = 0; i < m_buffer_size; i++) {
        score[c][i + (reading_midi - 1) * m_buffer_size] = 0;
        if (DEBUG) cout << i + (reading_midi - 1) * m_buffer_size << endl;
      }
    }
    return true; // done
  }
  double barStart = 0;
  double barEnd = 240 / tempo_map[0].second * timesig_map[0].second;
  // fill with midi data: hit, vel, tempo, timesig, pos_in_bar

  int counter = 0;
  int i = startFrom; // read from here
  int writeTo;       // write to here
  while (counter < m_buffer_size) {
    if (i >= midifile[1].size()) {
      break;
    }
    while(!midifile[1][i].isNoteOn() && i < midifile[1].size() - 1) {
      i++;
      skip++;
    }

    // tempo changes
    while ((current_tempo_index < tempo_map.size() - 1) && (midifile[1][i].tick >= tempo_map[current_tempo_index+1].first)) {
      current_tempo_index++;
    }
    // time signature changes
    while ((current_timesig_index < timesig_map.size() - 1) && (midifile[1][i].tick >= timesig_map[current_timesig_index+1].first)) {
      current_timesig_index++;
    }
    // bar positions
    if (midifile[1][i].seconds >= barEnd * 0.999) {
      barStart = barEnd;
      barEnd += 240 / tempo_map[current_tempo_index].second * timesig_map[current_timesig_index].second;
    }    
    double pos_in_bar = (midifile[1][i].seconds - barStart) / (barEnd - barStart);

    if (DEBUG) cout << midifile[1][i].seconds
        << ' ' << int(midifile[1][i][1])
        << ' ' << tempo_map[current_tempo_index].second
        << ' ' << timesig_map[current_timesig_index].second
        << ' ' << pos_in_bar
        << endl;

    writeTo = counter + (reading_midi - 1) * m_buffer_size;

    score[0][writeTo] = midifile[1][i][1]; // hit
    score[1][writeTo] = midifile[1][i][2]; // vel
    score[2][writeTo] = tempo_map[current_tempo_index].second; // tempo
    score[3][writeTo] = timesig_map[current_timesig_index].second; // timesig
    score[4][writeTo] = pos_in_bar; // pos_in_bar
    score[5][writeTo] = midifile[1][i].seconds * 1000.; // ms
    counter++;
    i++;
  }
  score_size = writeTo + 1;
  int upTo = reading_midi * m_buffer_size + skip;
  if (upTo >= midifile[1].size()) {
    return true; // done
  }
  return false; // not done yet
}

void rolypoly::prepareToPlay() {
  if (m_compute_thread && m_compute_thread->joinable()) {
    cout << "JOINING THREAD" << endl;
    m_compute_thread->join();
    cout << "JOINED THREAD" << endl;
  }
  playhead_ms = i_toModel = i_fromModel =
    t_score = t_play = 0;
  play_notes.clear();
  play_notes.reserve(score_size);
}

void rolypoly::playMidiIntoVector() {
  // populate vector of arrays with notes that don't have a tau yet
  // taking all the notes in the upcoming lookahead_ms
  double start_ms = playhead_ms; //score[TIME_MS][i_toModel];
  //cout << "taking new notes looking ahead from " << start_ms << " ms" << endl;
  in_notes.clear();
  in_notes.reserve(lookahead_ms / 100); // 10 notes per second
  if (m_model.get_attribute_as_string("generate") == "false") {
    double timestep_ms = score[TIME_MS][i_toModel];
    // get all notes in the next lookahead_ms
    while (timestep_ms < start_ms + lookahead_ms && i_toModel < score_size) {
      std::array<double, IN_DIM> note;
      for (int c = 0; c < IN_DIM; c++) {
        note[c] = score[c][i_toModel];
      }
      in_notes.push_back(note);
      i_toModel++;
      if (DEBUG) cout << "added timestep " << i_toModel << " at " << timestep_ms << " ms" << endl;    
      timestep_ms = score[TIME_MS][i_toModel];
    }
  } // TODO: "generate" == "true" -> play latest note from play_notes
    // TODO: if pos_in_bar < previous pos_in_bar, then we have a new bar
}

void rolypoly::vectorToModel(std::vector<std::array<double, IN_DIM>> &v) {
  // in: vector v (1, len, IN_DIM=5)
  // out: tensor modelOut (1, OUT_DIM=14, len)
  int length = v.size();

  // create a tensor from the vector
  torch::Tensor input_tensor = torch::zeros({1, IN_DIM, length});
  for (int c = 0; c < IN_DIM; c++) {
    for (int i = 0; i < length; i++) {
      input_tensor[0][c][i] = v[i][c];
    }
  }
  cout << "input_tensor:" << input_tensor << endl;
  // send the onset to the model
  modelOut = m_model.get_model().forward({input_tensor}).toTensor();
  cout << "output:" << modelOut << endl;
  // print the output shape
}

void rolypoly::getTauFromModel() {
  // populate play_notes[...i_toModel][TAU]
  long writeTo = i_fromModel;
  int i = 0;
  while (writeTo < i_toModel) {
    play_notes.emplace_back(std::array<double, OUT_DIM>());
    for (int c = 0; c < modelOut.size(1); c++) {
      play_notes[writeTo][c] = modelOut[0][c][i].item<double>();
      if (DEBUG && c==TAU) cout << writeTo << " got tau " << play_notes[writeTo][TAU] << endl;
      writeTo++; i++;
    }
  }
  i_fromModel = writeTo;
}

double rolypoly::computeNextNoteTimeMs() {
  if (!done_playing) { // m_model.get_attribute_as_string("generate") == "false" &&
    return score[TIME_MS][t_score] + play_notes[t_play][TAU];
  } else {
    // TODO: "generate" == "true" -> use latest notes from play_notes
  }
  return 0;
}

void rolypoly::incrementPlayIndexes() {
  // increment t_score and t_play
  double current_time_ms = score[TIME_MS][t_score];
  if (DEBUG) cout << "current note time: " << current_time_ms << endl;
  while (score[TIME_MS][t_score] == current_time_ms) {
    t_score++;
  }
  t_play++;
}

void rolypoly::processLiveOnsets(audio_bundle input) {
  int location = -1;
  for (int i = 0; i < input.frame_count(); i++) {
    // find onset
    if (input.samples(0)[i]) {
      location = i;
      break;
    }
  }
  if (location == -1) return; // no onset in this buffer

  // get the onset time in ms
  double onset_time_ms = playhead_ms + 
    lib::math::samples_to_milliseconds(location - latency, samplerate());
  
  cout << "-> ONSET at " << onset_time_ms << " ms" << endl;

  // find the closest note in the score
  int closest_note = 0;
  double closest_note_time = score[TIME_MS][0];
  for (int i = 0; i < t_score+1; i++) { // for all notes played so far
    double note_time = score[TIME_MS][i];
    if (abs(note_time - onset_time_ms) < abs(closest_note_time - onset_time_ms)) {
      closest_note = i;
      closest_note_time = note_time;
    }
  }

  // is the onset within 1/3 of the closest note duration?
  double closest_note_duration;
  if (onset_time_ms > closest_note_time && closest_note < score_size-1) {
    closest_note_duration = score[TIME_MS][closest_note+1] - closest_note_time;
  } else if (onset_time_ms < closest_note_time && closest_note > 0) {
    closest_note_duration = closest_note_time - score[TIME_MS][closest_note-1];
  } else return;

  double tau_guitar = onset_time_ms - closest_note_time;
  if (abs(tau_guitar) < closest_note_duration/3) {
    // if so, then we have a hit
    cout << "closest note is " << closest_note << " at " << closest_note_time << " ms" << endl;
  } else return;

  torch::Tensor input_tensor = torch::zeros({1, IN_DIM, 1});
  input_tensor[0][0][0] = 666; // mark this as a live onset
  input_tensor[0][1][0] = tau_guitar; // in ms
  for (int c = 2; c < IN_DIM; c++) {
    input_tensor[0][c][0] = score[c][closest_note]; // bpm, tsig, pos_in_bar
  }
  //cout << "input: " << input_tensor << endl;
  // send the onset to the model
  auto output = m_model.get_model().forward({input_tensor}).toTensor();
  cout << "output: " << output << endl;
}

void rolypoly::operator()(audio_bundle input, audio_bundle output) {
  // CHECK IF MODEL IS LOADED AND ENABLED
  if (!m_model.is_loaded() || !enable) {
    fill_with_zero(output);
    return;
  }

  perform(input, output);
}

void rolypoly::perform(audio_bundle input, audio_bundle output) {
  auto vec_size = input.frame_count();
  // INPUT
  if (m_model.get_attribute_as_string("play") == "true") {
    processLiveOnsets(input);
  }

  // OUTPUT
  if (m_model.get_attribute_as_string("play") == "true") {
    // if the "play" attribute is true,
    // if there are notes to play, play them
    
    // increment playhead
    double buf_ms = lib::math::samples_to_milliseconds(vec_size, samplerate());
    double next_ms; // usually it's the next note that doesn't have a tau yet
    // but at the end of the score, it can be the final note (computeNext...)
    next_ms = std::max(score[TIME_MS][i_toModel], computeNextNoteTimeMs() );
    if (playhead_ms < next_ms)
      playhead_ms += buf_ms;

    //if (DEBUG) cout << playhead_ms << " " << computeNextNoteTimeMs() << endl;

    if (playhead_ms >= computeNextNoteTimeMs() - buf_ms && !done_playing) {
      // when the time comes, play the microtime-adjusted note
      int micro_index = (computeNextNoteTimeMs() - playhead_ms) / buf_ms * vec_size;
      for (int c = 0; c < output.channel_count(); c++) {
        auto out = output.samples(c);
        for (int i = 0; i < output.frame_count(); i++) {
          out[i] = 0.;
        }
        out[micro_index] = play_notes[t_play][c];
      }
      incrementPlayIndexes();
    } else {
      // if the time hasn't come yet, do nothing
      fill_with_zero(output);
    }
    if (playhead_ms >= midifile[1].back().seconds * 1000. || t_score >= score_size) {
      if (DEBUG) cout << "reached end of midifile" << endl;
      attr = "play"; attr_value = "false"; set_attr();
      done_playing = true;
      timer_mode = TIMER::INACTIVE;
      m_timer.stop();
      if (m_compute_thread && m_compute_thread->joinable()) {
        cout << "==END==JOINING THREAD" << endl;
        m_compute_thread->join();
        cout << "==END==JOINED THREAD" << endl;
      }
    }
  } else {
    fill_with_zero(output);
  }
};

MIN_EXTERNAL(rolypoly);