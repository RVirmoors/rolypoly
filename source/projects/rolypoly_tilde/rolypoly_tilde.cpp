// Rolypoly C++ implementation
// 2023 rvirmoors
// v2.0.1 initially based on nn~ by Antoine Caillon & Axel Chemla-Romeu-Santos
//
// Frontend: Max external object

#define DEBUG true

#ifndef VERSION
#define VERSION "2.0.1"
#endif

// midi stuff
#include "MidiFile.h"

// Max & Torch
#include "c74_min.h"
#include "torch/torch.h"
#include "backend.hpp"

// C++ stuff
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

double bartime_to_ms(double from_pos, double to_pos, double bpm, double tsig) {
  while (to_pos < from_pos)
    to_pos += 1.0;
  double bartime = to_pos - from_pos;
  return bartime * 1000. / 60. * bpm * tsig;
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

  // OPERATION MODES
  bool m_read;
  bool m_play;
  bool m_train;
  bool m_use_thread;

  // SCORE RELATED MEMBERS
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
  at::Tensor play_notes; // hits to be played
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

  // ATTRIBUTES
  attribute<bool> enable{this, "enable", true,
                         description{"Enable / disable tensor computation"}};

  attribute<int> latency{this, "latency", 512,
                         description{"Onset detection latency (samples)"}};

  attribute<bool> generate{this, "generate", false,
                         description{"Generate hits on the fly, not reading from the score"}};

  attribute<bool> score_filter{this, "score_filter", true,
                         description{"Filter out notes not in the score"}};

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
      torch::Tensor input_tensor = torch::randn({1, 1, INPUT_DIM}).to(device);
      for (int i = 0; i < 7; i++) {
          auto start = std::chrono::high_resolution_clock::now();
          try {
              torch::NoGradGuard no_grad_guard;
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
    if (DEBUG) cout << "== M_TIMER == play_ms  | size  :  " << playhead_ms << " | " << score.size(0) << endl;
    if (timer_mode == TIMER::READ) {
      //cout << "timer read" << endl;
      read_deferred.set();
    } else if (timer_mode == TIMER::PLAY) {
      cout << "timer play" << endl;
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
        if (DEBUG) cout << "started thread" << endl;
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
          config.epochs = 20;
          backend::finetune(model, config, score, play_notes, m_follow, device);
        }
        catch (std::exception& e)
        {
            cerr << e.what() << endl;
        }
        torch::save(model, "roly_fine.pt");
        cout << "Done. Saved roly_fine.pt" << endl;
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
      if (!score.size(0)) {
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
      if (!score.size(0)) {
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
  torch::load(model, path, device);
  cout << "Loaded finetuned model" << endl;
  model->eval();
}

void rolypoly::initialiseScore() {
  score = torch::zeros({0, INPUT_DIM}).to(device);
  score_ms.clear();
}

rolypoly::rolypoly(const atoms &args)
    : m_compute_thread(nullptr), m_loaded(false),
      m_read(false), m_play(false), m_train(false),
      m_use_thread(true), lookahead_ms(500), m_follow(0.4) {

  if (torch::cuda::is_available()) {
      cout << "Using CUDA." << endl;
      device = torch::kCUDA;
  } else {
    cout << "No CUDA found, using CPU." << endl;
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
    h_m_path = get_latest_model("roly_hits.pt");
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
    torch::load(model, m_path, device);
    torch::load(hitsModel, h_m_path, device);
  } catch (std::exception& e)
  {
      if (DEBUG) cerr << e.what() << endl;
      cout << "Error loading models." << endl;
  }
  m_loaded = true;

  // LOAD FINETUNED MODEL IF EXISTS
  try {
    loadFinetuned("roly_fine.pt");
  }         
  catch (std::exception& e)
  {
      // if (DEBUG) cerr << e.what() << endl;
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

  int i = 0;      // note index in midi (a hit can have multiple notes)
  double prevTime = -1.;
  at::Tensor hit = torch::zeros({1, INPUT_DIM}).to(device);
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
        score = torch::cat({score, hit}, 0);
        score_ms.push_back(prevTime * 1000.); // ms
        assert(score.size(0) == score_ms.size());
      }
      prevTime = midifile[1][i].seconds;
      hit = torch::zeros({1, INPUT_DIM}).to(device);
    }
    hit[0][pitch_class_map[midifile[1][i][1]]] = midifile[1][i][2]; // hit, vel
    hit[0][INX_BPM] = tempo_map[current_tempo_index].second; // tempo
    hit[0][INX_TSIG] = timesig_map[current_timesig_index].second; // timesig
    hit[0][INX_BAR_POS] = pos_in_bar; // pos_in_bar

    i++;
  }

  if (DEBUG) cout << "sent " << score.size(0) << " hits to model" << endl;

  if (DEBUG) cout << score.index({Slice(0,5)}) << endl;
  
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
  play_notes = torch::zeros({0, INPUT_DIM}).to(device);
}

void rolypoly::advanceReadHead() {
  // advance t_toModel over all the notes in the upcoming lookahead_ms
  if (DEBUG) cout << "== MID2VEC == looking ahead from " << playhead_ms << " ms" << endl;
  if (DEBUG) cout << "t_play: " << t_play << " | t_toModel: " << t_toModel << endl;
  // get all notes in the next lookahead_ms
  while (score_ms[t_toModel] < playhead_ms + lookahead_ms && t_toModel < score.size(0) - 1) {
    t_toModel++;
  }
}

void rolypoly::tensorToModel() {
  // read from score up to t_toModel, and return the model output
  // in: tensor score (1, ...t_toModel, INPUT_DIM=22)
  // out: tensor modelOut (1, ...t_toModel, OUTPUT_DIM=18)
  // if in generate mode, then run the HitsModel to insert a new note into the score

  int newNotes = t_toModel - t_fromModel;
  torch::Tensor modelOut = torch::zeros({1, 1, OUTPUT_DIM}).to(device);

  if (!newNotes) { // no new notes
    return;
  }

  if (newNotes > BLOCK_SIZE-1) {
    cout << "WARNING: more than " << BLOCK_SIZE-1 << " notes need processing. Only considering the latest " << BLOCK_SIZE-1 << ".";
    while (newNotes > BLOCK_SIZE - 1) {
      // copy notes from score into play_notes
      play_notes = torch::cat({play_notes, score[t_fromModel].unsqueeze(0)}, 0);
      t_fromModel++;                      
      newNotes--;
    }
  }

  long start = std::max(0, t_toModel - BLOCK_SIZE);
  cout << "sending " << start << " - " << t_toModel << endl;
  torch::Tensor input_tensor = score.index({Slice(start, t_toModel)}).unsqueeze(0);
  backend::dataScaleDown(input_tensor);

  // if (DEBUG) cout << "== VEC2MOD == input_tensor  :  " << input_tensor << endl;

  // send the notes to the model
  try {
    torch::NoGradGuard no_grad_guard;
    at::set_num_threads(1)           // Disables the intraop thread pool.
    at::set_num_interop_threads(1). // Disables the interop thread pool.
    modelOut = model(input_tensor).detach_();
    // modelOut = input_tensor.index({Slice(), Slice(0, input_tensor.size(1)), Slice(0, input_tensor.size(2))});
    backend::dataScaleUp(input_tensor);
    backend::dataScaleUp(modelOut);
    // if (DEBUG) cout << "== VEC2MOD == output  :  " << modelOut << endl;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  // populate play_notes[...t_toModel]
  if (DEBUG) cout << "== TAUfromMOD == notes from model: " << t_toModel << "-" << t_fromModel << "=" << newNotes << endl;

  for (int i = modelOut.size(1) - newNotes; i < modelOut.size(1); i++) {
    // TODO if use score velocities, just take 9-18 from modelOut
    torch::Tensor new_note;
    if (score_filter) {
      new_note = torch::cat({
        input_tensor[0][i].index({Slice(0, 9)}),
        modelOut[0][i].index({Slice(9, 18)})
          });
    }        
    else
      new_note = modelOut[0][i].index({Slice(0, OUTPUT_DIM - 1)});

    new_note = torch::cat({new_note, input_tensor[0][i].index({Slice(INX_BPM, INX_TAU_G)})});
    new_note = torch::cat({new_note, modelOut[0][i][18].unsqueeze(0)}); // last output channel: tau_g_hat
    new_note.unsqueeze_(0);

    play_notes = torch::cat({play_notes, new_note}, 0);

    // cout << "play_notes is now: " << play_notes.index({Slice(), Slice(9, 18)}) << endl;
  }

  
  if (generate) {
      /*
      cout << "GENERATE" << endl;
    // compute a new note to be played next = inserted into the score at t_toModel
    input_tensor = score.index({Slice(start, t_toModel)}).unsqueeze(0);
    backend::dataScaleDown(input_tensor);
    torch::NoGradGuard no_grad_guard;
    modelOut = hitsModel(input_tensor);
    backend::dataScaleUpHits(modelOut);
    int last = modelOut.size(1) - 1;
    cout << modelOut << endl;
    
    torch::Tensor generated_note = torch::cat({
      modelOut[0][last].index({Slice(1, 10)}), // hits generated
      torch::zeros({9}).to(device), // offsets estimated above, to be filled in below
      score[t_toModel].index({Slice(INX_BPM, INX_BAR_POS)}), // tempo & t_sig from prev note
      modelOut[0][last][0].unsqueeze(0), // bar pos generated
      torch::zeros({1}).to(device), // tau_guitar, to be filled in on onset detect
    });

    cout << generated_note << endl;
    score = torch::cat({
      score.index({Slice(0, t_toModel)}),
      generated_note.unsqueeze(0),
      score.index({Slice(t_toModel, None)})
    }, 0);
    double generated_note_ms = score_ms[t_toModel] + 
      bartime_to_ms(score[t_toModel][INX_BAR_POS].item<double>(), 
                        generated_note[INX_BAR_POS].item<double>(), 
                        generated_note[INX_BPM].item<double>(),
                        generated_note[INX_TSIG].item<double>());
    score_ms.insert(score_ms.begin() + t_toModel, generated_note_ms);
    assert(score.size(0) == score_ms.size());
    */
  }

  // copy executed offsets to score (to be later fed into the model for inference)
  if (t_toModel < score.size(0) - 1)
    score.index_put_({Slice(t_fromModel+1, t_toModel+1), Slice(9, 18)}, 
      play_notes.index({Slice(t_fromModel, t_toModel), Slice(9, 18)}));

  t_fromModel = t_toModel;
}

std::pair<double, int> rolypoly::computeNextNoteTimeMs() {
  if (!done_playing) { 
    if (t_play >= play_notes.size(0)) {
      cout << "no tau yet: " << t_play << " > " << play_notes.size(0) << endl;
      return std::make_pair(score_ms[t_play], -1);
    }
    // find next earliest hit in play_notes[t_play]
    // which hasn't been played yet ( > played_ms )
    // if all hits have been played, increment t_play and look again
    double earliest_ms = std::numeric_limits<double>::infinity();
    int earliest_channel = -1;
    for (int c = 9; c < 18; c++) {
      if (play_notes[t_play][c-9].item<double>() > 0.0) { // if vel > 0
        double this_ms = score_ms[t_play] + play_notes[t_play][c].item<double>();
        if (this_ms < earliest_ms && this_ms > played_ms) {
          earliest_ms = this_ms;
          earliest_channel = c;
        }
      }
    }
    if (earliest_channel != -1) {
      // cout << "earliest channel: " << earliest_channel << endl;
      return std::make_pair(earliest_ms, earliest_channel - 9);
    }
    else {
      t_play++;
      cout << "next note: " << t_play << endl;
      return computeNextNoteTimeMs();
    }
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
  if (onset_time_ms > closest_note_time && closest_note < score.size(0)-1) {
    int next_note = closest_note+1;
    while (score_ms[next_note] == closest_note_time && next_note < score.size(0)-1) next_note++;
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
  score[closest_note][INX_TAU_G] = tau_guitar;
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
  if (m_play && play_notes.size(0) && !done_playing) {
    processLiveOnsets(input);
  }

  // OUTPUT
  if (m_play) {
    // if the "play" attribute is true,
    fill_with_zero(output);

    if (!play_notes.size(0)) {
      // no notes yet!
      return;
    }
    // increment playhead
    double bufsize_ms = lib::math::samples_to_milliseconds(vec_size, samplerate());
    std::pair<double, int> next_note = computeNextNoteTimeMs(); // usually it's the next note that doesn't have a tau yet

    // but at the end of the score, it can be the final note (computeNext...)
    next_note.first = std::min(score_ms[t_toModel], next_note.first );
    if (playhead_ms < next_note.first)
      playhead_ms += bufsize_ms;

    if (next_note.second == -1) {
      // no tau yet
      return;
    }

    // if (DEBUG) cout << "playhead: " << playhead_ms << " | next note @ " << next_note.first << endl;
    while (playhead_ms >= next_note.first - bufsize_ms && !done_playing) {
      // when the time comes, play the microtime-adjusted note
      long micro_index = (next_note.first - playhead_ms) / bufsize_ms * vec_size;
      micro_index = std::min(micro_index, vec_size-1);
      micro_index = std::max(micro_index, 0L);
      auto out = output.samples(next_note.second);
      double vel = play_notes[t_play][next_note.second].item<double>();
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
      played_ms = next_note.first;
      next_note = computeNextNoteTimeMs();
      //next_note.first = std::max(score_ms[t_toModel], next_note.first );
    }

    // cout << "end is at " << midifile[1].back().seconds * 1000. << " | " << score.size(0) << endl;

    if (playhead_ms >= midifile[1].back().seconds * 1000. || t_play >= score.size(0) - 1) {
      cout << "Done playing. To finetune the model based on this run, send the 'train' message." << endl;
      m_play = false;
      done_playing = true;
      timer_mode = TIMER::INACTIVE;
      m_timer.stop();
      if (m_compute_thread && m_compute_thread->joinable()) {
        //if (DEBUG) cout << "== END == JOINING THREAD" << endl;
        m_compute_thread->join();
        if (DEBUG) cout << "== END == JOINED THREAD" << endl;
      }
    }
  } else {
    fill_with_zero(output);
  }
};

MIN_EXTERNAL(rolypoly);