// 2023 rvirmoors
// based on nn~ by Antoine Caillon & Axel Chemla-Romeu-Santos

#define DEBUG false

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

c74::min::path get_latest_model(std::string model_path) {
  if (model_path.substr(model_path.length() - 3) != ".ts")
    model_path = model_path + ".ts";
  // namespace max = c74::max;
  // char          m_filename[max::MAX_PATH_CHARS];
  // max::t_fourcc m_type;
  // short         m_path;
  // strncpy(m_filename, model_path.c_str(), max::MAX_PATH_CHARS);
  // auto err = max::locatefile_extended(m_filename, &m_path, &m_type, NULL, 0);
  return path(model_path);
}

// function to read a tensor and return a csv string
std::string tensor_to_csv(at::Tensor tensor) {
  // in: tensor of shape (length, channels)
  // out: csv string of shape (length, channels)
  std::string csv = "";
  for (int i = 0; i < tensor.size(0); i++) {
    for (int j = 0; j < tensor.size(1); j++) {
      csv += std::to_string(tensor[i][j].item<double>());
      if (j < tensor.size(1) - 1)
        csv += ",";
    }
    csv += "\n";
  }
  return csv;
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

  // ATTRIBUTES
  bool m_read;
  bool m_play;
  bool m_generate;
  bool m_train;

  // MIDI RELATED MEMBERS
  MidiFile midifile;
  c74::min::path m_midi_path;
  std::vector<std::array<double, SCORE_DIM>> score;
  long i_toModel; // next timestep to be sent to the model
  long t_score; // next timestep to be played from score
  int reading_midi;
  int skip; // used to skip everything but NoteOn events
  bool done_reading;

  // PLAY RELATED MEMBERS
  double playhead_ms;  // in ms
  long i_fromModel; // next timestep to be read from the model  
  long t_play; // next timestep to be played from play_notes
  long last_onset; // last timestep with an onset detected
  std::vector<std::array<double, IN_DIM>> in_notes; // notes to be sent to the model
  std::array<double, IN_DIM> in_onset; // onset to be sent to the model
  at::Tensor modelOut; // result from calling model.forward()
  std::vector<std::array<double, OUT_DIM>> play_notes; // hits to be played
  bool done_playing;
  int lookahead_ms; // in ms
  short timer_mode; // 0 inactive, 1 read, 2 play
  enum TIMER {INACTIVE, READ, PLAY, TRAIN};

  std::vector<std::pair<long, double>> tempo_map;
  int current_tempo_index;
  std::vector<std::pair<long, double>> timesig_map;
  int current_timesig_index;
  double barStart = 0, barEnd = 0;

  double m_follow; // [0 ... 1] how much to finetune the model to follow the guitar

  void loadFinetuned(std::string path);
  void initialiseScore();
  void parseTimeEvents(MidiFile &midifile);
  bool midiNotesToModel();

  void prepareToPlay();
  void playMidiIntoVector();
  void vectorToModel(std::vector<std::array<double, IN_DIM>> &v);
  void getTauFromModel();
  double computeNextNoteTimeMs();
  bool incrementPlayIndexes();
  void processLiveOnsets(audio_bundle input);

	// BACKEND RELATED MEMBERS
	Backend m_model;
	std::string m_method;
	std::vector<std::string> settable_attributes;
	bool has_settable_attribute(std::string attribute);
	c74::min::path m_path;
  int m_in_dim, m_out_dim;

	// AUDIO PERFORM
  bool m_use_thread;
	std::unique_ptr<std::thread> m_compute_thread;
	void operator()(audio_bundle input, audio_bundle output);
	void buffered_perform(audio_bundle input, audio_bundle output);
	void perform(audio_bundle input, audio_bundle output);


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

  attribute<bool> score_filter{this, "score_filter", true,
                         description{"Filter out notes not in the score"}}; // TODO: implement

  // BOOT STAMP
  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        cout << "rolypoly~ v" << VERSION << " - 2023 Grigore Burloiu - rvirmoors.github.io" << endl;
        cout << "based on nn~ by Antoine Caillon & Axel Chemla-Romeu-Santos" << endl;
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
      torch::Tensor input_tensor = torch::randn({1, 1, IN_DIM});
      for (int i = 0; i < 7; i++) {
          auto start = std::chrono::high_resolution_clock::now();
          try {
              m_model.get_model().forward({ input_tensor }).toTensor();
          }
          catch (std::exception& e)
          {
              cerr << e.what() << endl;
          }
          auto end = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
          if (DEBUG) cout << "step " << i+1 << "/7: " << duration.count() / 1000. << " ms" << endl;
      }
      cout << "Done. A model run lasts around " << int(duration.count() / 1000.) << " ms." << endl;
      // if the model is too slow, we need to increase the lookahead
      if (duration.count() / 1000. > lookahead_ms / 4) {
        lookahead_ms = duration.count() / 1000. * 6 + 50;
        cout << "That's too slow. Increasing lookahead to " << lookahead_ms << " ms." << endl;
      }
      return {};
    }
  };

  timer<timer_options::defer_delivery> m_timer { this, MIN_FUNCTION {
    if (DEBUG) cout << "== M_TIMER == play_ms | t_score | size  :  " << playhead_ms << " | " << t_score << " | " << score.size() << endl;
    if (timer_mode == TIMER::READ) {
      //cout << "timer read" << endl;
      read_deferred.set();
    } else if (timer_mode == TIMER::PLAY) {
      //cout << "timer play" << endl;
      perform_threaded.set();
      if (!done_playing) {
        m_timer.delay(lookahead_ms / 4);
      }
    } else if (timer_mode == TIMER::TRAIN) {
      //cout << "timer train" << endl;
      train_deferred.set();
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
      }
      // make a tensor from the score
      torch::Tensor score_tensor = torch::zeros({1, int(score.size()), IN_DIM});
      for (int i = 0; i < score.size(); i++) {
        for (int j = 0; j < IN_DIM; j++) {
          score_tensor[0][i][j] = score[i][j];
        }
      }
      m_model.get_model().forward({score_tensor}).toTensor();

      if (done_reading && reading_midi) {
        cout << "Done reading the score." << endl;
        //cout << modelOut << endl;
        reading_midi = 0;
        attr = "read"; attr_value = "false"; set_attr();
        m_read = false;
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
        //if (DEBUG) cout << "joining - performing " << playhead_ms << endl;
        // get any available model outputs from the previous run
        m_compute_thread->join();
        if (DEBUG) cout << "joined at " << playhead_ms << " ms"<< endl;
      }

      // send midi notes
      // to the in_notes buffer, to later get the model output
      if (!done_playing)
        playMidiIntoVector();

      // run the model on the in_notes buffer
      if (m_use_thread && !done_playing && in_notes.size()) {
        m_compute_thread = std::make_unique<std::thread>(&rolypoly::vectorToModel, this, in_notes);
        //if (DEBUG) cout << "started thread" << endl;
      }
      return {};
    }
  };

  queue<> train_deferred {this, 
    MIN_FUNCTION {
      if (DEBUG) cout << "train_deferred" << endl;
      if (m_train) {
        //m_model.get_model().save("model_pre.pt");
        cout << "Finetuning the model... this could take a while." << endl;
        torch::AutoGradMode enable_grad(true);
        m_model.get_model().train();
        // send ones to train & get loss
        torch::Tensor input_tensor = torch::ones({1, 1, IN_DIM});
        input_tensor[0][0][1] = m_follow;
        torch::Tensor losses;
        try {
            losses = m_model.get_model().forward({ input_tensor }).toTensor();
        }
        catch (std::exception& e)
        {
            cerr << e.what() << endl;
        }
        cout << "Done. Losses:\nTOTAL   Dhat-D   Vhat-V  Dhat-G  Ghat-G\n" << losses.slice(2, 0, 5) << endl;
        // save model
        m_model.get_model().save("model.pt");
        cout << "Saved model.pt" << endl;
        loadFinetuned("model.pt");
        if (DEBUG) {
          // send zeros to get diag info
          input_tensor = torch::zeros({1, 1, IN_DIM});
          torch::Tensor diag;
          try {
              diag = m_model.get_model().forward({ input_tensor }).toTensor();
          }
          catch (std::exception& e)
          {
              cerr << e.what() << endl;
          }
          cout << "D_hat     G_hat     G\n" << diag.slice(2, 0, 3).slice(1, 0, 10) << endl;
        }
        // reset the training flag
        m_train = false;
        attr = "finetune"; attr_value = "false"; set_attr();
      }
      return {};
    }
  };

  message<> read {this, "read", "Load score",
    MIN_FUNCTION {
      done_reading = false;
      initialiseScore();
      attr = "read"; attr_value = "true"; set_attr();
      m_read = true;
      timer_mode = TIMER::READ;
      m_timer.delay(0);
      return {};
    }
  };

  message<> start {this, "start", "Start playing",
    MIN_FUNCTION {
      if (!score.size()) {
        cerr << "no score loaded, can't play yet!" << endl;
        return {};
      }
      done_playing = false;
      prepareToPlay();
      attr = "play"; attr_value = "true"; set_attr();
      m_play = true;
      timer_mode = TIMER::PLAY;
      m_timer.delay(0);
      
      return {};
    }
  };

  message<> train {this, "train", "Finetune the model based on the latest run",
    MIN_FUNCTION {
      if (!score.size()) {
        cerr << "no score loaded, can't train yet!" << endl;
        return {};
      }
      if (args.size() == 1) {
        m_follow = args[0];
        cout << "Finetuning with Follow = " << m_follow << endl;
      } else {
        cout << "Using default Follow: " << m_follow << endl;
      }

      attr = "finetune"; attr_value = "true"; set_attr();
      m_train = true;
      timer_mode = TIMER::TRAIN;
      m_timer.delay(50);
      
      return {};
    }
  };
};

void rolypoly::loadFinetuned(std::string path) {
  torch::jit::script::Module finetuned = torch::jit::load(path);
  // copy the parameters from the finetuned model to the base model
  torch::AutoGradMode enable_grad(false);
  std::unordered_map<std::string, at::Tensor> name_to_tensor;
  for (auto& param : finetuned.named_parameters()) {
      name_to_tensor[param.name] = param.value;
  }
  for (auto& base_param : m_model.get_model().named_parameters()) {
      auto name = base_param.name;
      auto& base_tensor = base_param.value;
      auto fine_tuned_param = name_to_tensor[name];
      base_tensor.copy_(fine_tuned_param);
      base_tensor.detach_();
      base_tensor.requires_grad_(true);
      // cout << "Loaded " << name << endl;
  }
  cout << "Loaded finetuned model" << endl;
  m_model.get_model().eval();
}

void rolypoly::initialiseScore() {
  score.clear();
  score.reserve(MAX_SCORE_LENGTH);
}

rolypoly::rolypoly(const atoms &args)
    : m_compute_thread(nullptr),
      m_read(false), m_play(false), m_generate(false), m_train(false),
      m_method("forward"),
      m_use_thread(true), lookahead_ms(500), m_follow(0.4) {

  m_model = Backend();

  // CHECK ARGUMENTS
  if (!args.size()) {
    return;
  }
  if (args.size() > 0) { // ONE ARGUMENT IS GIVEN
    auto model_path = std::string(args[0]);
    m_path = get_latest_model(model_path);
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

  // TRY TO LOAD MODEL
  if (m_model.load(std::string(m_path))) {
      cerr << "error during loading" << endl;
      error();
      return;
  }

  // LOAD FINETUNED MODEL IF EXISTS
  try {
    loadFinetuned("model.pt");
  }         
  catch (std::exception& e)
  {
      if (DEBUG) cerr << e.what() << endl;
      cout << "No finetuned model found." << endl;
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

  // Calling forward in a thread causes memory leak in windows.
  // See https://github.com/pytorch/pytorch/issues/24237
//#ifdef _WIN32
//  m_use_thread = false;
//#endif

  // CREATE INLET, OUTLETS
  m_inlets.push_back(std::make_unique<inlet<>>(
    this, "(signal) musician input", "signal"));

  for (int i(0); i < m_out_dim; i++) {
    std::string output_label = "";
    try {
      output_label = m_model.get_model().attr(m_method + "_output_labels").toList().get(i).toStringRef();
    } catch (...) {
      output_label = "(signal) model output " + std::to_string(i);
    }
    m_outlets.push_back(std::make_unique<outlet<>>(
        this, output_label, "signal"));
  }
  cout << "Running warmup, please wait (Max will freeze for a few seconds) ..." << endl;
  // "play must be set to true in the python module for this to work"
  warmup.delay(500);
}

rolypoly::~rolypoly() {
  if (m_compute_thread && m_compute_thread->joinable())
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

bool rolypoly::midiNotesToModel() {
  // populates score with midi data: hit, vel, tempo, timesig, pos_in_bar

  int counter = 0;// hit index
  int i = 0;      // note index in midi (a hit can have multiple notes)

  while (i < midifile[1].size()) {

    // skip non-noteOn events
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
    if (barEnd == 0) // initialise
      barEnd = 240 / tempo_map[0].second * timesig_map[0].second;
    if (midifile[1][i].seconds >= barEnd * 0.99) {
      barStart = barEnd;
      barEnd += 240 / tempo_map[current_tempo_index].second * timesig_map[current_timesig_index].second;
      //cout << "EOB" << endl;
    }
    double pos_in_bar = std::max(0., midifile[1][i].seconds - barStart) / (barEnd - barStart);

    if (DEBUG) cout << midifile[1][i].seconds
        << ' ' << int(midifile[1][i][1])
        << ' ' << tempo_map[current_tempo_index].second
        << ' ' << timesig_map[current_timesig_index].second
        << ' ' << pos_in_bar
        << ' ' << endl;

    std::array<double, 6> hit;
    hit[0] = midifile[1][i][1]; // hit
    hit[1] = midifile[1][i][2]; // vel
    hit[2] = tempo_map[current_tempo_index].second; // tempo
    hit[3] = timesig_map[current_timesig_index].second; // timesig
    hit[4] = pos_in_bar; // pos_in_bar
    hit[5] = midifile[1][i].seconds * 1000.; // ms
    score.emplace_back(hit);

    counter++;
    i++;
  }

  if (DEBUG) cout << "sent " << counter << " == " << score.size() << " hits to model" << endl;
  
  if (i >= midifile[1].size()) {
    return true; // done
  }
  return false; // not done yet (SHOULD NEVER HAPPEN)
}

void rolypoly::prepareToPlay() {
  if (m_compute_thread && m_compute_thread->joinable()) {\
    m_compute_thread->join();
  }
  playhead_ms = i_toModel = i_fromModel =
    t_score = t_play = 0;
  play_notes.clear();
  play_notes.reserve(score.size());
  
  modelOut = torch::zeros({1, 1, OUT_DIM});
}

void rolypoly::playMidiIntoVector() {
  // populate vector of arrays with notes that don't have a tau yet
  // taking all the notes in the upcoming lookahead_ms
  double start_ms = playhead_ms; //score[i_toModel][TIME_MS];
  if (DEBUG) cout << "== MID2VEC == looking ahead from " << start_ms << " ms" << endl;
  in_notes.clear();
  in_notes.reserve(lookahead_ms / 100); // 10 notes per second
  if (!m_generate) {
    double timestep_ms = score[i_toModel][TIME_MS];
    // get all notes in the next lookahead_ms
    while (timestep_ms < start_ms + lookahead_ms && i_toModel < score.size()) {
      std::array<double, IN_DIM> note;
      for (int c = 0; c < IN_DIM; c++) {
        note[c] = score[i_toModel][c];
      }
      in_notes.push_back(note);      
      if (DEBUG) cout << "== MID2VEC == score_i_toModel | score_ms  :  " << i_toModel << " | " << timestep_ms << " ms" << endl;
      i_toModel++;
      timestep_ms = score[i_toModel][TIME_MS];
    }
  } // TODO: "generate" == "true" -> play latest note from play_notes
    // TODO: if pos_in_bar < previous pos_in_bar, then we have a new bar
}

void rolypoly::vectorToModel(std::vector<std::array<double, IN_DIM>> &v) {
  // in: vector v (1, len, IN_DIM=5)
  // out: tensor modelOut (1, len, OUT_DIM=14)
  int length = v.size();
  if (!length) {
    modelOut = torch::zeros({1, 1, OUT_DIM});
    return;
  }
  
  if (v[0][2] < 1) { // note with BPM is zero (should never happen)
    modelOut = torch::zeros({1, 1, OUT_DIM});
    cout << "HUH WHAT" << endl;
    return;    
  }

  // create a tensor from the vector
  torch::Tensor input_tensor = torch::zeros({1, length, IN_DIM});
  for (int c = 0; c < IN_DIM; c++) {
    for (int i = 0; i < length; i++) {
      input_tensor[0][i][c] = v[i][c];
    }
  }
 // if (DEBUG) cout << "== VEC2MOD == input_tensor  :  " << input_tensor << endl;
  // send the notes to the model
  if (m_model.is_loaded()) {
    try {
      modelOut = m_model.get_model().forward({input_tensor}).toTensor();
      // if (DEBUG) cout << "== VEC2MOD == output  :  " << modelOut << endl;
    } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
    }
  }
  getTauFromModel();
}

void rolypoly::getTauFromModel() {
  // populate play_notes[...i_toModel][TAU]
  if (modelOut[0][0][9].item<double>() < 0.1) {
    // not ready to play yet (THIS SHOULD NEVER HAPPEN)
    if (DEBUG) cout << "== TAUfromMOD == zero bpm from model" << endl;
    return;
  }
  long writeTo = i_fromModel;
  int i = 0;
  if (DEBUG) cout << "== TAUfromMOD == notes from model: " << modelOut.size(1) << endl;

  while (i < modelOut.size(1)) {
    play_notes.emplace_back(std::array<double, OUT_DIM>());
    for (int c = 0; c < modelOut.size(2); c++) {
      play_notes[writeTo][c] = modelOut[0][i][c].item<double>();
      if (DEBUG && c==TAU) cout << "== TAUfromMOD == play_note " << writeTo << " got tau: " << play_notes[writeTo][TAU] << endl;
    }
    writeTo++; i++;
  }
  i_fromModel = writeTo; // next time continue from here
}

double rolypoly::computeNextNoteTimeMs() {
  if (!m_generate && !done_playing) { 
    if (t_play >= play_notes.size()) {
      //cout << "no tau yet" << endl;
      return score[t_score][TIME_MS];
    }
    return score[t_score][TIME_MS] + play_notes[t_play][TAU];
  } else {
    // TODO: "generate" == "true" -> use latest notes from play_notes
  }
  return 0;
}

bool rolypoly::incrementPlayIndexes() {
  // increment t_score and t_play
  double current_time_ms = score[t_score][TIME_MS];
  if (DEBUG) cout << "== PERFORM == just played: " << t_play << " | " << score[t_score][0] << " | " << current_time_ms << "+" << play_notes[t_play][TAU] << " ms" << endl;
  while (score[t_score][TIME_MS] == current_time_ms) {
    t_score++;
    if (t_score >= score.size()) {
      return true; // done
    }
  }
  t_play++;
  return false; // not done
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
  
  if (DEBUG) cout << "== ONSET == at " << onset_time_ms << " ms" << endl;

  // find the closest note in the score
  int closest_note = 0;
  double closest_note_time = score[0][TIME_MS];
  for (int i = 0; i < t_score+1; i++) { // for all notes played so far
    double note_time = score[i][TIME_MS];
    if (abs(note_time - onset_time_ms) < abs(closest_note_time - onset_time_ms)) {
      closest_note = i;
      closest_note_time = note_time;
    }
  }
  if (closest_note == last_onset) return; // don't send the same note twice
  last_onset = closest_note;

  // is the onset within 1/3 of the closest note duration?
  double closest_note_duration;
  if (onset_time_ms > closest_note_time && closest_note < score.size()-1) {
    int next_note = closest_note+1;
    while (score[next_note][TIME_MS] == closest_note_time && next_note < score.size()-1) next_note++;
    closest_note_duration = score[next_note][TIME_MS] - closest_note_time;
  } else if (onset_time_ms < closest_note_time && closest_note > 0) {
    int prev_note = closest_note-1;
    while (score[prev_note][TIME_MS] == closest_note_time && prev_note > 0) prev_note--;
    closest_note_duration = closest_note_time - score[prev_note][TIME_MS];
  } else return; // no duration to compare to

  double tau_guitar = onset_time_ms - closest_note_time;
  if (abs(tau_guitar) < closest_note_duration/3) {
    // if so, then we have a hit
    if (DEBUG) cout << "closest note is " << closest_note << " at " << closest_note_time << " ms" << endl;
  } else {
    if (DEBUG) cout << "NOT within " << closest_note_duration/3 << " of " << closest_note_time << endl; return;
  }


  torch::Tensor input_tensor = torch::zeros({1, 1, IN_DIM});
  input_tensor[0][0][0] = 666; // mark this as a live onset
  input_tensor[0][0][1] = tau_guitar; // in ms
  for (int c = 2; c < IN_DIM; c++) {
    input_tensor[0][0][c] = score[closest_note][c]; // bpm, tsig, pos_in_bar
  }
  // send the onset to the model
  // if (DEBUG) {
      try {
          auto output = m_model.get_model().forward({ input_tensor }).toTensor();
          // cout << "ONSET x_dec:\n" << output << endl;
      } catch (std::exception& e)
      {
          cerr << e.what() << endl;
      }
  // } else {
  //   m_model.get_model().forward({input_tensor}).toTensor();
  // }
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
  if (m_play && play_notes.size() && !done_playing) {
    processLiveOnsets(input);
  }

  // OUTPUT
  if (m_play) {
    // if the "play" attribute is true,
    if (!play_notes.size()) {
      // no notes yet!
      fill_with_zero(output);
      return;
    }
    // increment playhead
    double buf_ms = lib::math::samples_to_milliseconds(vec_size, samplerate());
    double next_ms; // usually it's the next note that doesn't have a tau yet
    // but at the end of the score, it can be the final note (computeNext...)
    next_ms = std::max(score[i_toModel][TIME_MS], computeNextNoteTimeMs() );
    if (playhead_ms < next_ms)
      playhead_ms += buf_ms;

    //if (DEBUG) cout << playhead_ms << " " << computeNextNoteTimeMs() << endl;
    fill_with_zero(output);
    while (playhead_ms >= computeNextNoteTimeMs() - buf_ms && !done_playing) {
      // when the time comes, play the microtime-adjusted note(s)
      long micro_index = (computeNextNoteTimeMs() - playhead_ms) / buf_ms * vec_size;
      micro_index = std::min(micro_index, vec_size-1);
      micro_index = std::max(micro_index, 0L);
      for (int c = 0; c < output.channel_count(); c++) {
        auto out = output.samples(c);
        double vel = play_notes[t_play][c];
        // cout << "vel: " << vel << " at t: " << t_play << endl;
        if (c < 9) {
          if (vel > 0)
            out[micro_index] = std::max(std::min(vel / 127., 1.), 0.1);
        }
        else
          out[micro_index] = vel;
      }
      bool done = incrementPlayIndexes();
      if (done) break;
    }
    if (playhead_ms >= midifile[1].back().seconds * 1000. || t_score >= score.size()) {
      cout << "Done playing. To finetune the model based on this run, send the 'train' message." << endl;
      attr = "play"; attr_value = "false"; set_attr();
      m_play = false;
      done_playing = true;
      timer_mode = TIMER::INACTIVE;
      m_timer.stop();
      if (m_compute_thread && m_compute_thread->joinable()) {
        if (DEBUG) cout << "== END == JOINING THREAD" << endl;
        m_compute_thread->join();
        if (DEBUG) cout << "== END == JOINED THREAD" << endl;
      }
    }
  } else {
    fill_with_zero(output);
  }
};

MIN_EXTERNAL(rolypoly);