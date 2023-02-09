// 2023 rvirmoors
// adapted from nn~ by Antoine Caillon & Axel Chemla-Romeu-Santos

#define DEBUG true

#ifndef VERSION
#define VERSION "2.0b1"
#endif

// midi stuff
#include "MidiFile.h"
#define MAX_SCORE_LENGTH 100000
#define SCORE_DIM 6 // hit, vel, bpm, tsig, pos_in_bar, time_sec
#define PLAY_DIM 13 // 9 vels, bpm, tsig, pos_in_bar, tau 
#define TIME_SEC 5
#define TAU 12

// nn~
#include "c74_min.h"
#include "torch/torch.h"
#include "../../../nn_tilde/src/backend/backend.h"
#include "../../../nn_tilde/src/frontend/maxmsp/shared/circular_buffer.h"
#include <string>
#include <thread>
#include <vector>

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
	MIN_DESCRIPTION {"Expressive Drum Machine: read MIDI file, listen to audio, and output drums"};
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
  long t_score; // next timestep to be sent to the model
  int reading_midi;  
  int skip; // used to skip everything but NoteOn events
  bool done_reading;

  double playhead;  // in ms
  long t_play; // next timestep to be played from live_notes
  std::vector<std::vector<double>> play_notes; // hits to be played
  bool done_playing;
  int m_lookahead_ms; // in ms

  std::vector<std::pair<long, double>> tempo_map;
  int current_tempo_index;
  std::vector<std::pair<long, double>> timesig_map;
  int current_timesig_index;

  void initialiseScore();
  void parseTimeEvents(MidiFile &midifile);
  void resetInputBuffer();
  bool midiNotesToModel();
  void prepareToPlay();
  void playMidiIntoModel();
  void getTauFromModel(long vec_size);
  double computeNextNoteTimeMs();

	// BACKEND RELATED MEMBERS
	Backend m_model;
	std::string m_method;
	std::vector<std::string> settable_attributes;
	bool has_settable_attribute(std::string attribute);
	c74::min::path m_path;
	int m_in_dim, m_in_ratio, m_out_dim, m_out_ratio, m_higher_ratio;

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

  timer<timer_options::defer_delivery> m_timer { this, MIN_FUNCTION {
    if (m_model.get_attribute_as_string("read") == "true") {
      cout << "timer read" << endl;
      read_deferred.set();
    } else if (m_model.get_attribute_as_string("play") == "true") {
      cout << "timer play" << endl;
      perform_threaded.set();
      if (!done_playing) {
        m_timer.delay(m_lookahead_ms);
      }
    }
    return {};
  }};

  queue<> read_deferred {this, 
    MIN_FUNCTION {
      resetInputBuffer();
      reading_midi ++;
      if (DEBUG) cout << reading_midi << " " << done_reading << endl;
      
      if (reading_midi && !done_reading) {
        // populate score and place it into m_in_buffer
        done_reading = midiNotesToModel();
        for (int c(0); c < m_in_dim; c++) {
          m_in_buffer[c].put(score[c], m_buffer_size);
        }
        if (DEBUG) cout << "input buffers read: " << reading_midi << endl;
      } else {
        // fill m_in_buffer with zeros
        for (int c(0); c < m_in_dim; c++) {
          m_in_buffer[c].reset();
        }
      }

      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_in_dim; c++)
        m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);

      // PROCESS SCORE
      model_perform();

      if (!done_reading) {
        m_timer.delay(10);
      }
      return {};
    }
  };

  queue<> perform_threaded { this,
    MIN_FUNCTION {
      if (m_compute_thread && m_use_thread) {
        if (DEBUG) cout << "joining - performing" << endl;
        if (DEBUG) cout << m_compute_thread->get_id() << endl;
        m_compute_thread->join();
      }
/*
      int n;
      while (nums.try_dequeue(n)) {
        cout << "got " << n << endl;
      }
*/
      if (m_in_buffer[0].full()) { // BUFFER IS FULL
        cout << "buffer full " << playhead << endl;

        // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
        for (int c(0); c < m_in_dim; c++)
          m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);

        // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
        for (int c(0); c < m_out_dim; c++)
          m_out_buffer[c].put(m_out_model[c].get(), m_buffer_size);

        if (done_reading && reading_midi) {
          if (DEBUG) cout << "done reading" << endl;
          reading_midi = 0;
          attr = "read"; attr_value = "false"; set_attr();
          prepareToPlay();
        }
        // TODO: what if the buffer is full of midi notes but not done_reading? Does reading several buffers work?
        cout << "done buffer full" << endl;
      }

      if (m_use_thread) {
        m_compute_thread = std::make_unique<std::thread>(&rolypoly::wait_a_sec, this);
        if (DEBUG) cout << "started thread" << endl;
        if (DEBUG) cout << m_compute_thread->get_id() << endl;
      }
      return {};
    }
  };

  message<> read {this, "read", "Load score",
    MIN_FUNCTION {
      done_reading = false;
      attr = "read"; attr_value = "true"; set_attr();
      m_timer.delay(10);
      return {};
    }
  };

  message<> start {this, "start", "Start playing",
    MIN_FUNCTION {
      done_playing = false;
      prepareToPlay();
      attr = "play"; attr_value = "true"; set_attr();
      m_timer.delay(10);
      return {};
    }
  };

};

void rolypoly::wait_a_sec() {
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

void rolypoly::model_perform() {
  std::vector<float *> in_model, out_model;
  for (int c(0); c < m_in_dim; c++)
    in_model.push_back(m_in_model[c].get());
  for (int c(0); c < m_out_dim; c++)
    out_model.push_back(m_out_model[c].get());
  m_model.perform(in_model, out_model, m_buffer_size, m_method, 1);
}

void rolypoly::initialiseScore() {
  score = new double*[SCORE_DIM];
  for (int c(0); c < SCORE_DIM; c++) {
    score[c] = new double[MAX_SCORE_LENGTH];
  }
}

rolypoly::rolypoly(const atoms &args)
    : m_compute_thread(nullptr), m_in_dim(1), m_in_ratio(1), m_out_dim(1),
      m_out_ratio(1), m_buffer_size(4096), m_method("forward"),
      m_use_thread(true), m_lookahead_ms(2000) {

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
    playhead = t_score = t_play =
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

  m_higher_ratio = m_model.get_higher_ratio();

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
  m_in_ratio = params[1];
  m_out_dim = params[2];
  m_out_ratio = params[3];

  if (!m_buffer_size) {
    // NO THREAD MODE
    //m_use_thread = false;
    m_buffer_size = m_higher_ratio;
  } else if (m_buffer_size < m_higher_ratio) {
    m_buffer_size = m_higher_ratio;
    cerr << "buffer size too small, switching to " << m_buffer_size << endl;
  } else {
    m_buffer_size = power_ceil(m_buffer_size);
  }

  if (m_buffer_size < 16) {
    cerr << "buffer size too small, should be at least 16" << endl;
    m_buffer_size = 16;
  }

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

  cout << "initial performance" << endl;
  model_perform();
  cout << "initial performance done" << endl;
}

rolypoly::~rolypoly() {
  if (m_compute_thread)
  m_compute_thread->join();
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

void rolypoly::resetInputBuffer() {
  if(!m_in_buffer[0].empty() && !reading_midi && !done_reading) {
    if (DEBUG) cout << "resettin" << endl;
    // if the buffer isn't empty, reset it
    for (int c(0); c < m_in_dim; c++)
      m_in_buffer[c].reset();
    done_reading = false;
    if (DEBUG) cout << "starting to read" << endl;
  }
}

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
  int i = startFrom;
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
      //cout << "== bar ==" << barStart << " " << barEnd << endl;
    }    
    double pos_in_bar = (midifile[1][i].seconds - barStart) / (barEnd - barStart);

    if (DEBUG) cout << midifile[1][i].seconds
        << ' ' << int(midifile[1][i][1])
        << ' ' << tempo_map[current_tempo_index].second
        << ' ' << timesig_map[current_timesig_index].second
        << ' ' << pos_in_bar
        << endl;

    int loc = counter + (reading_midi - 1) * m_buffer_size;

    score[0][loc] = midifile[1][i][1]; // hit
    score[1][loc] = midifile[1][i][2]; // vel
    score[2][loc] = tempo_map[current_tempo_index].second; // tempo
    score[3][loc] = timesig_map[current_timesig_index].second; // timesig
    score[4][loc] = pos_in_bar; // pos_in_bar
    score[5][loc] = midifile[1][i].seconds * 1000.; // ms
    counter++; 
    i++;
  }
  int upTo = reading_midi * m_buffer_size + skip;
  if (upTo >= midifile[1].size()) {
    return true; // done
  }
  return false; // not done yet
}

void rolypoly::prepareToPlay() {
  playhead = t_score = t_play = 0;
  play_notes.clear();
  std::vector<double> first_note = {0,0,0,0,0,0,0,0,0,
    score[2][0], score[3][0], score[4][0], 0}; // placeholder
  play_notes.push_back(first_note);
}

void rolypoly::playMidiIntoModel() {
  // populate m_in_buffer with notes that don't have a tau yet
  double tau = play_notes[t_play][TAU];
  if (!tau) {
    // send a timestep
    if (m_model.get_attribute_as_string("generate") == "false") {
      double timestep_ms = score[TIME_SEC][t_score];
      int t_send; // temporary variable, for processing simultaneous hits
      for (int c = 0; c < m_in_dim; c++) {
        t_send = t_score;
        double *in = new double[m_buffer_size];
        for (int i = 0; i < m_buffer_size; i++) {
          // get all notes at the timestep
          if (timestep_ms == score[TIME_SEC][t_send]) {
            in[i] = score[c][t_send];
            t_send++;
          } else {
          // the rest stays zero
          in[i] = 0;
          }
        }
        m_in_buffer[c].put(in, m_buffer_size);
      }
      if (DEBUG) cout << "sent timestep " << t_score << " at " << timestep_ms << " ms" << endl;
      t_score = t_send; // t_send has been incremented for simultaneous notes
    } // TODO: "generate" == "true" -> play latest note from play_notes
  } else {
    for (int c = 0; c < m_in_dim; c++)
      m_in_buffer[c].reset(); // empty buffer, don't run model.perform yet
  }
}

void rolypoly::getTauFromModel(long vec_size) {
  // populate play_notes[...t_play][TAU]
  std::vector<double> new_hit;
  double* out = new double[vec_size];
  for (int c = 0; c < m_out_dim; c++) {
    m_out_buffer[c].get(out, vec_size);
    new_hit.push_back(out[0]);
  }
  play_notes[t_play] = new_hit;
  if (DEBUG) cout << "new tau " << new_hit[TAU] << endl;
  std::vector<double> next_note = {0,0,0,0,0,0,0,0,0,
    score[2][t_score], score[3][t_score], score[4][t_score], 0};
  play_notes.push_back(next_note);
}

double rolypoly::computeNextNoteTimeMs() {
  double buf_ms = lib::math::samples_to_milliseconds(m_buffer_size, samplerate());
  if (m_model.get_attribute_as_string("generate") == "false") {
    return score[TIME_SEC][t_score] + play_notes[t_play][TAU];
  } else {
    // TODO: "generate" == "true" -> use latest note from play_notes
  }
  return 0;
}

void rolypoly::operator()(audio_bundle input, audio_bundle output) {
  auto dsp_vec_size = output.frame_count();

  // CHECK IF MODEL IS LOADED AND ENABLED
  if (!m_model.is_loaded() || !enable) {
    fill_with_zero(output);
    return;
  }

  // CHECK IF DSP_VEC_SIZE IS LARGER THAN BUFFER SIZE
  if (dsp_vec_size > m_buffer_size) {
    cerr << "vector size (" << dsp_vec_size << ") ";
    cerr << "larger than buffer size (" << m_buffer_size << "). ";
    cerr << "disabling model.";
    cerr << endl;
    enable = false;
    fill_with_zero(output);
    return;
  }

  perform(input, output);
}

void rolypoly::perform(audio_bundle input, audio_bundle output) {
  auto vec_size = input.frame_count();

  // POPULATING INPUT CIRCULAR BUFFER
  if (m_model.get_attribute_as_string("play") == "true") {
    // if the "play" attribute is true, send midi notes
    // to the circular buffer, to later get the model output
    playMidiIntoModel();
  } else {
    // neither reading, nor playing: do nothing
    fill_with_zero(output);
    return;
  }

  // OUTPUT
  if (m_model.get_attribute_as_string("play") == "true") {
    // if the "play" attribute is true,
    // if model just performed, get the model output
    //cout << "play " << endl;
    if (!m_out_buffer[0].empty() 
      && play_notes.size() == t_play + 1) {
        //getTauFromModel(vec_size);
    }
    // if there are notes to play, play them
    
    double buf_ms = lib::math::samples_to_milliseconds(vec_size, samplerate());
    playhead += buf_ms;
    if (DEBUG) cout << t_play << " " << playhead << " " << computeNextNoteTimeMs() << endl;
    if (playhead >= midifile[1].back().seconds * 1000.) {
      if (DEBUG) cout << "reached end of midifile" << endl;
      attr = "play"; attr_value = "false"; set_attr();
      done_playing = true;
      prepareToPlay();
      fill_with_zero(output);
      return;
    }
    if (false) { //(play_notes[t_play][TAU] && playhead >= computeNextNoteTimeMs() - buf_ms && !done_playing) {
      // when the time comes, play the microtime-adjusted note
      int micro_index = (computeNextNoteTimeMs() - playhead) / buf_ms * vec_size;
      for (int c = 0; c < output.channel_count(); c++) {
        auto out = output.samples(c);
        for (int i = 0; i < output.frame_count(); i++) {
          out[i] = 0.;
        }
        out[micro_index] = play_notes[t_play][c];
      }
      t_play++;
    } else {
      // if the time hasn't come yet, do nothing
      fill_with_zero(output);
    }
  } else {
      fill_with_zero(output);
  }
};

MIN_EXTERNAL(rolypoly);