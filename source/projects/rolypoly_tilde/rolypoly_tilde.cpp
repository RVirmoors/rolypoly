// Rolypoly C++ implementation
// 2023 rvirmoors
// v2.0b1 initially based on nn~ by Antoine Caillon & Axel Chemla-Romeu-Santos
//
// Frontend: Max external object

#define DEBUG true

#ifndef VERSION
#define VERSION "2.0b2"
#endif

// midi stuff
#include "MidiFile.h"
#define MAX_SCORE_LENGTH 100000

// Max & Torch
#include "c74_min.h"
#include "torch/torch.h"
#include "backend.hpp"
#include <string>
#include <thread>
#include <vector>
#include <chrono>

using namespace c74::min;
using namespace smf;
using namespace at::indexing;

// ======= useful functions ==============

c74::min::path get_latest_model(std::string model_path) {
  if (model_path.substr(model_path.length() - 3) != ".pt")
    model_path = model_path + ".pt";
  return path(model_path);
}

void fill_with_zero(audio_bundle output) {
  for (int c(0); c < output.channel_count(); c++) {
    auto out = output.samples(c);
    for (int i(0); i < output.frame_count(); i++) {
      out[i] = 0.;
    }
  }
}

// ========= MAIN ROLYPOLY~ CLASS =========

class rolypoly : public object<rolypoly>, public vector_operator<> {
public:
	MIN_DESCRIPTION {"Expressive Drum Machine: read MIDI file, listen to audio, output drums"};
	MIN_TAGS {"drums, sync, deep learning"};
	MIN_AUTHOR {"Grigore Burloiu // rvirmoors"};
	MIN_RELATED {"nn~, flucoma~"};

	// INLETS OUTLETS
	std::vector<std::unique_ptr<inlet<>>> m_inlets;
	std::vector<std::unique_ptr<outlet<>>> m_outlets;
  atoms m_note;
  // kick, snar, hcls, hopn, ltom, mtom, htom, cras, ride
  const int playableNotes[9] = {36, 38, 42, 46, 43, 45, 48, 57, 51};

	rolypoly(const atoms &args = {});
	~rolypoly();

  // ATTRIBUTES
  bool m_read;
  bool m_play;
  bool m_generate;
  bool m_train;
  bool m_use_thread;

  // MIDI RELATED MEMBERS
  MidiFile midifile;
  c74::min::path m_midi_path;
  at::Tensor score;
  vector<double> score_ms;
  int t_toModel; // next timestep to be sent to the model
  int reading_midi;
  int skip; // used to skip everything but NoteOn events
  bool done_reading;

  // PLAY RELATED MEMBERS
  double playhead_ms;  // in ms
  double played_ms; // latest offset-adjusted note that has been played
  int t_fromModel; // next timestep to be read from the model  
  int t_play; // next timestep to be played from play_notes
  int last_onset; // last timestep with an onset detected
  at::Tensor modelOut; // result from calling model->forward()
  std::vector<std::array<double, INPUT_DIM>> play_notes; // hits to be played
  bool done_playing;
  int lookahead_ms; // in ms
  int timer_mode; // 0 inactive, 1 read, 2 play
  enum TIMER {INACTIVE, READ, PLAY, TRAIN};

  std::vector<std::pair<long, double>> tempo_map;
  int current_tempo_index;
  std::vector<std::pair<long, double>> timesig_map;
  int current_timesig_index;
  double barStart = 0, barEnd = 0;

  double m_follow; // [0 ... 1] how much to direct the model to follow the guitar

  void loadFinetuned(std::string path);
  void initialiseScore();
  void parseTimeEvents(MidiFile &midifile);
  bool midiNotesToScore();

  void prepareToPlay();
  void advanceReadHead();
  void tensorToModel();
  std::pair<double, int> computeNextNoteTimeMs();
  void processLiveOnsets(audio_bundle input);

	// BACKEND RELATED MEMBERS
  torch::Device device = torch::kCPU;
  backend::TransformerModel model = nullptr;//(INPUT_DIM, OUTPUT_DIM, 128, 16, 12, 12);
  backend::HitsTransformer hitsModel = nullptr;//(128, 16, 12);
	c74::min::path m_path, h_m_path;  
  bool m_loaded;

	// AUDIO PERFORM
	std::unique_ptr<std::thread> m_compute_thread;
	void operator()(audio_bundle input, audio_bundle output);
	void perform(audio_bundle input, audio_bundle output);


  // ONLY FOR DOCUMENTATION
  argument<symbol> path_arg{this, "model path",
                            "Absolute path to the pretrained model."};
  argument<symbol> hit_path_arg{this, "hit_model path",
                            "Absolute path to the hit generation model."}; // TODO: implement

  // ENABLE / DISABLE ATTRIBUTE
  attribute<bool> enable{this, "enable", true,
                         description{"Enable / disable tensor computation"}};

  // LATENCY (guitar onsets) ATTRIBUTE
  attribute<int> latency{this, "latency", 512,
                         description{"Onset detection latency (samples)"}};

  attribute<bool> score_filter{this, "score_filter", true,
                         description{"Filter out notes not in the score"}}; // TODO: implement

  attribute<bool> signal_out{this, "signal_out", true,
                         description{"Output signals"}};

  attribute<bool> message_out{this, "message_out", true,
                         description{"Output messages"}};

  // BOOT STAMP
  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        cout << "rolypoly~ v" << VERSION << " - 2023 Grigore Burloiu - rvirmoors.github.io" << endl;
        return {};
      }};

  timer<timer_options::defer_delivery> warmup { this,
    MIN_FUNCTION {
      // warmup the model
      std::chrono::microseconds duration;
      torch::Tensor input_tensor = torch::randn({1, 1, INPUT_DIM});
      for (int i = 0; i < 7; i++) {
          auto start = std::chrono::high_resolution_clock::now();
          try {
              model->forward(input_tensor);
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
    if (DEBUG) cout << "== M_TIMER == play_ms  | size  :  " << playhead_ms << " | " << score.size(1) << endl;
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
        // populate score
        initialiseScore();
        done_reading = midiNotesToScore();
      }

      if (done_reading && reading_midi) {
        cout << "Done reading the score." << endl;
        reading_midi = 0;
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

      // look for notes up to lookahead_ms
      if (!done_playing)
        advanceReadHead();

      // run the model on notes found
      if (m_use_thread && !done_playing) {
        m_compute_thread = std::make_unique<std::thread>(&rolypoly::tensorToModel, this);
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
        try {
          backend::TrainConfig config;
          config.block_size = BLOCK_SIZE;
          backend::finetune(model, config, score, play_notes, m_follow, device);
        }
        catch (std::exception& e)
        {
            cerr << e.what() << endl;
        }
        //cout << "Done. Losses:\nTOTAL   Dhat-D   Vhat-V  Dhat-G  Ghat-G\n" << losses.slice(2, 0, 5) << endl;
        // save model
        torch::save(model, "model.pt");
        cout << "Saved model.pt" << endl;
        // loadFinetuned("model.pt");
        // if (DEBUG) {
        //   // send zeros to get diag info
        //   input_tensor = torch::zeros({1, 1, INPUT_DIM});
        //   torch::Tensor diag;
        //   try {
        //       diag = model->forward(input_tensor);
        //   }
        //   catch (std::exception& e)
        //   {
        //       cerr << e.what() << endl;
        //   }
        //   cout << "D_hat     G_hat     G\n" << diag.slice(2, 0, 3).slice(1, 0, 10) << endl;
        // }
        // reset the training flag
        m_train = false;
        //enable_grad(false);
      }
      return {};
    }
  };

  message<> read {this, "read", "Load score",
    MIN_FUNCTION {
      done_reading = false;
      initialiseScore();
      m_read = true;
      timer_mode = TIMER::READ;
      m_timer.delay(0);
      return {};
    }
  };

  message<> start {this, "start", "Start playing",
    MIN_FUNCTION {
      if (!score.size(1)) {
        cerr << "no score loaded, can't play yet!" << endl;
        return {};
      }
      done_playing = false;
      prepareToPlay();
      m_play = true;
      timer_mode = TIMER::PLAY;
      m_timer.delay(0);
      
      return {};
    }
  };

  message<> train {this, "train", "Finetune the model based on the latest run",
    MIN_FUNCTION {
      if (!score.size(1)) {
        cerr << "no score loaded, can't train yet!" << endl;
        return {};
      }
      if (args.size() == 1) {
        m_follow = args[0];
        cout << "Finetuning with Follow = " << m_follow << endl;
      } else {
        cout << "Using default Follow: " << m_follow << endl;
      }

      m_train = true;
      timer_mode = TIMER::TRAIN;
      m_timer.delay(50);
      
      return {};
    }
  };
};

void rolypoly::loadFinetuned(std::string path) {
  torch::load(model, path);
  cout << "Loaded finetuned model" << endl;
  model->eval();
}

void rolypoly::initialiseScore() {
  score = torch::zeros({1, 0, INPUT_DIM});
  score_ms.clear();
}

rolypoly::rolypoly(const atoms &args)
    : m_compute_thread(nullptr), m_loaded(false),
      m_read(false), m_play(false), m_generate(false), m_train(false),
      m_use_thread(true), lookahead_ms(500), m_follow(0.4) {

  if (torch::cuda::is_available()) {
      cout << "Using CUDA." << endl;
      device = torch::kCUDA;
  }
  model = backend::TransformerModel(INPUT_DIM, OUTPUT_DIM, 128, 16, 12, 12, device);
  hitsModel = backend::HitsTransformer(128, 16, 12, device);

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
    playhead_ms = t_fromModel = t_toModel = 
      played_ms = t_play =
      current_tempo_index = current_timesig_index = 
      skip = 0;
  }

  // TRY TO LOAD MODELS
  try {
    torch::load(model, m_path);
    torch::load(hitsModel, "roly_hits.pt");
  } catch (std::exception& e)
  {
      if (DEBUG) cerr << e.what() << endl;
      cout << "Error loading models." << endl;
  }
  m_loaded = true;

  // LOAD FINETUNED MODEL IF EXISTS
  try {
    loadFinetuned("model.pt");
  }         
  catch (std::exception& e)
  {
      if (DEBUG) cerr << e.what() << endl;
      cout << "No finetuned model found." << endl;
  }

  // Calling forward in a thread causes memory leak in windows.
  // See https://github.com/pytorch/pytorch/issues/24237
//#ifdef _WIN32
//  m_use_thread = false;
//#endif

  // CREATE INLET, OUTLETS
  m_inlets.push_back(std::make_unique<inlet<>>(
    this, "(signal) musician input", "signal"));

  std::array<std::string, 9> labels = {"K", "S", "Hcl", "Hop", "lT", "mT", "hT", "Cr", "Rd"};

  for (std::string output_label : labels) {
    m_outlets.push_back(std::make_unique<outlet<>>(
        this, output_label, "signal"));
  }

  m_outlets.push_back(std::make_unique<outlet<>>(
    this, "(list) notes out"));
  m_note.reserve(2); // note, velocity

  cout << "Running warmup, please wait (Max will freeze for a few seconds) ..." << endl;
  // "play must be set to true in the python module for this to work"
  warmup.delay(500);
}

rolypoly::~rolypoly() {
  if (m_compute_thread && m_compute_thread->joinable())
    m_compute_thread->join();
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

bool rolypoly::midiNotesToScore() {
  // populates score with midi data: hit, vel, tempo, timesig, pos_in_bar

  int counter = 0;// hit index TODO remove, this is used just for debug
  int i = 0;      // note index in midi (a hit can have multiple notes)
  double prevTime = -1.;
  at::Tensor hit = torch::zeros({1, 1, INPUT_DIM});
  auto pitch_class_map = backend::classes_to_map();

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

    if (midifile[1][i].seconds > prevTime) {
      // new hit
      if (prevTime != -1.) {
        // unless this is the very first hit, add prev hit to score
        score = torch::cat({score, hit}, 1);
        score_ms.push_back(prevTime * 1000.); // ms
        assert(score.size(1) == score_ms.size());
      }
      prevTime = midifile[1][i].seconds;
      hit = torch::zeros({1, 1, INPUT_DIM});
    }
    hit[pitch_class_map[midifile[1][i][1]]] = midifile[1][i][2]; // hit, vel
    hit[INX_BPM] = tempo_map[current_tempo_index].second; // tempo
    hit[INX_TSIG] = timesig_map[current_timesig_index].second; // timesig
    hit[INX_BAR_POS] = pos_in_bar; // pos_in_bar

    counter++;
    i++;
  }

  if (DEBUG) cout << "sent " << counter << " == " << score.size(1) << " hits to model" << endl;
  
  if (i >= midifile[1].size()) {
    return true; // done
  }
  return false; // not done yet (SHOULD NEVER HAPPEN)
}

void rolypoly::prepareToPlay() {
  if (m_compute_thread && m_compute_thread->joinable()) {\
    m_compute_thread->join();
  }
  playhead_ms = t_toModel = t_fromModel =
    played_ms = t_play = 0;
  play_notes.clear();
  play_notes.reserve(score.size(1));
  
  modelOut = torch::zeros({1, 1, OUTPUT_DIM});
}

void rolypoly::advanceReadHead() {
  // advance t_toModel over all the notes in the upcoming lookahead_ms
  double start_ms = playhead_ms;
  if (DEBUG) cout << "== MID2VEC == looking ahead from " << start_ms << " ms" << endl;
  if (!m_generate) {
    double timestep_ms = score_ms[t_toModel];
    // get all notes in the next lookahead_ms
    while (timestep_ms < start_ms + lookahead_ms && t_toModel < score.size(1)) {
      t_toModel++;
      timestep_ms = score_ms[t_toModel];
    }
  } // TODO: "generate" == "true" -> play latest note from play_notes
    // TODO: if pos_in_bar < previous pos_in_bar, then we have a new bar
}

void rolypoly::tensorToModel() {
  // read from score up to t_toModel, and return the model output
  // in: tensor score (1, ...t_toModel, INPUT_DIM=22)
  // out: tensor modelOut (1, ...t_toModel, OUTPUT_DIM=18)

  int newNotes = t_toModel - t_fromModel;

  if (!newNotes) { // no new notes
    modelOut = torch::zeros({1, 1, OUTPUT_DIM});
    return;
  }

  if (newNotes > BLOCK_SIZE-1) {
    cout << "WARNING: more than " << BLOCK_SIZE-1 << " notes need processing. Only considering the latest " << BLOCK_SIZE-1 << ".";
    while (newNotes > BLOCK_SIZE - 1) {
      // copy notes from score into play_notes
      play_notes.emplace_back(std::array<double, INPUT_DIM>());
      for (int c = 0; c < modelOut.size(2); c++) {
        play_notes[t_fromModel][c] = score[0][t_fromModel][c].item<double>();
      }
      play_notes[t_fromModel][INX_BPM] = score[0][t_fromModel][INX_BPM].item<double>();
      play_notes[t_fromModel][INX_TSIG] = score[0][t_fromModel][INX_TSIG].item<double>();
      play_notes[t_fromModel][INX_BAR_POS] = score[0][t_fromModel][INX_BAR_POS].item<double>();
      play_notes[t_fromModel][INX_TAU_G] = 0.;
      t_fromModel++;                      
      newNotes--;
    }
  }

  long start = std::max(0, t_toModel - BLOCK_SIZE);
  torch::Tensor input_tensor = score.index({Slice(), Slice(start, t_toModel)});

  if (DEBUG) cout << "== VEC2MOD == input_tensor  :  " << input_tensor << endl;
  // send the notes to the model
  try {
    modelOut = model->forward(input_tensor);
    if (DEBUG) cout << "== VEC2MOD == output  :  " << modelOut << endl;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  // populate play_notes[...t_toModel]
  if (DEBUG) cout << "== TAUfromMOD == notes from model: " << newNotes << endl;

  for (int i = BLOCK_SIZE - newNotes; i > BLOCK_SIZE; i++) {
    play_notes.emplace_back(std::array<double, INPUT_DIM>());
    for (int c = 0; c < OUTPUT_DIM - 1; c++) {
      play_notes[t_fromModel][c] = modelOut[0][i][c].item<double>();
    }
    play_notes[t_fromModel][INX_BPM] = input_tensor[0][i-1][INX_BPM].item<double>();
    play_notes[t_fromModel][INX_TSIG] = input_tensor[0][i-1][INX_TSIG].item<double>();
    play_notes[t_fromModel][INX_BAR_POS] = input_tensor[0][i-1][INX_BAR_POS].item<double>();
    play_notes[t_fromModel][INX_TAU_G] = modelOut[0][i][18].item<double>(); // last output channel: tau_g_hat
    t_fromModel++;
  }
}

std::pair<double, int> rolypoly::computeNextNoteTimeMs() {
  if (!m_generate && !done_playing) { 
    if (t_play >= play_notes.size()) {
      cout << "no tau yet" << endl;
      return std::make_pair(score_ms[t_play], -1);
    }
    // find next earliest hit in play_notes[t_play]
    // if all hits have been played, increment t_play and look again
    double earliest_ms = std::numeric_limits<double>::infinity();
    int earliest_channel = -1;
    for (int c = 9; c < 18; c++) {
      double this_ms = score_ms[t_play] + play_notes[t_play][c];
      if (this_ms < earliest_ms && play_notes[t_play][c-9] > 0) {
        earliest_ms = this_ms;
        earliest_channel = c;
      }
    }
    if (earliest_channel != -1)
      return std::make_pair(earliest_ms, earliest_channel);
    else {
      t_play++;
      return computeNextNoteTimeMs();
    }
  } else {
    // TODO: "generate" == "true" -> use latest notes from play_notes
  }
  return std::make_pair(0, 0);
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
  double closest_note_time = score_ms[0];
  for (int i = 0; i <= t_play; i++) { // for all notes played so far
    double note_time = score_ms[i];
    if (abs(note_time - onset_time_ms) < abs(closest_note_time - onset_time_ms)) {
      closest_note = i;
      closest_note_time = note_time;
    }
  }
  if (closest_note == last_onset) return; // don't send the same note twice
  last_onset = closest_note;

  // is the onset within 1/3 of the closest note duration?
  double closest_note_duration;
  if (onset_time_ms > closest_note_time && closest_note < score.size(1)-1) {
    int next_note = closest_note+1;
    while (score_ms[next_note] == closest_note_time && next_note < score.size(1)-1) next_note++;
    closest_note_duration = score_ms[next_note] - closest_note_time;
  } else if (onset_time_ms < closest_note_time && closest_note > 0) {
    int prev_note = closest_note-1;
    while (score_ms[prev_note] == closest_note_time && prev_note > 0) prev_note--;
    closest_note_duration = closest_note_time - score_ms[prev_note];
  } else return; // no duration to compare to

  double tau_guitar = onset_time_ms - closest_note_time;
  if (abs(tau_guitar) < closest_note_duration/3) {
    // if so, then we have a hit
    if (DEBUG) cout << "closest note is " << closest_note << " at " << closest_note_time << " ms" << endl;
  } else {
    if (DEBUG) cout << "NOT within " << closest_note_duration/3 << " of " << closest_note_time << endl; return;
  }

  // add detected tau_g to score
  score[0][closest_note][INX_TAU_G] = tau_guitar;
}

void rolypoly::operator()(audio_bundle input, audio_bundle output) {
  // CHECK IF MODEL IS LOADED AND ENABLED
  if (!m_loaded || !enable) {
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
    std::pair<double, int> next_note = computeNextNoteTimeMs(); // usually it's the next note that doesn't have a tau yet
    // but at the end of the score, it can be the final note (computeNext...)
    next_note.first = std::max(score_ms[t_toModel], next_note.first );
    if (playhead_ms < next_note.first)
      playhead_ms += buf_ms;

    //if (DEBUG) cout << playhead_ms << " " << next_note << endl;
    fill_with_zero(output);
    while (playhead_ms >= next_note.first - buf_ms && !done_playing) {
      // when the time comes, play the microtime-adjusted note
      long micro_index = (next_note.first - playhead_ms) / buf_ms * vec_size;
      micro_index = std::min(micro_index, vec_size-1);
      micro_index = std::max(micro_index, 0L);
      auto out = output.samples(next_note.second);
      double vel = play_notes[t_play][next_note.second];
      // cout << "vel: " << vel << " at t: " << t_play << endl;
      if (vel > 0) {
        if (signal_out) // OUTPUT NOTE SIGNALS
          out[micro_index] = std::max(std::min(vel / 127., 1.), 0.1);
        if (message_out) { // OUTPUT MESSAGES
          int msg_index = output.channel_count();
          m_note[0] = playableNotes[next_note.second];
          m_note[1] = vel;
          m_outlets[msg_index].get()->send(m_note[0], m_note[1]);
        }
      }   
      next_note = computeNextNoteTimeMs();
      next_note.first = std::max(score_ms[t_toModel], next_note.first );
    }

    if (playhead_ms >= midifile[1].back().seconds * 1000. || t_play >= score.size(1)) {
      cout << "Done playing. To finetune the model based on this run, send the 'train' message." << endl;
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