// Rolypoly C++ implementation
// 2023 rvirmoors
// v2.0.1 initially based on nn~ by Antoine Caillon & Axel Chemla-Romeu-Santos
//
// Frontend: Max external object

#define DEBUG false

#ifndef VERSION
#define VERSION "2.0.1"
#endif

#define INX_DUMPOUT 10

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

unsigned power_ceil(unsigned x) {
  // from nn_tilde
  if (x <= 1)
    return 1;
  int power = 2;
  x--;
  while (x >>= 1)
    power <<= 1;
  return power;
}

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
  while (to_pos <= from_pos)
    to_pos += 1.0;
  double bartime = to_pos - from_pos;
  return bartime * 1000. / 60. * bpm * tsig;
}

torch::Tensor bartime_to_ms(torch::Tensor bartime, torch::Tensor bpm, torch::Tensor tsig) {
  return bartime * 1000. / 60. * bpm * tsig;
}

torch::Tensor ms_to_bartime(double ms, torch::Tensor bpm, torch::Tensor tsig) {
  return ms / 1000. * 60. / bpm / tsig;
}

torch::Tensor ms_to_bartime(torch::Tensor ms, torch::Tensor bpm, torch::Tensor tsig) {
  return ms / 1000. * 60. / bpm / tsig;
}

// ============= MAIN ROLYPOLY~ CLASS =============

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
  const int out_pitches[9] = {36, 38, 42, 46, 43, 45, 48, 57, 51};

	rolypoly(const atoms &args = {});
	~rolypoly();

  // OPERATION
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
  double upTo_ms; // generate notes up to this timepoint
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

  torch::Tensor train_ops; // 3 [0..1] values: vel, follow, predict

  void dumpout(string prefix, double value);

  torch::Tensor finetuneLoss(torch::Tensor out, torch::Tensor x);
  torch::Tensor finetune(backend::TrainConfig config);
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
  argument<symbol> path_arg{this, "midi path",
                            "Path to the midi drum score file (.mid)"};

  // ATTRIBUTES
  attribute<bool> enable{this, "enable", true,
                         description{"Enable / disable tensor computation"}};

  attribute<int> latency{this, "latency", 512,
                         description{"Onset detection latency (samples)"}};

  attribute<bool> generate{this, "generate", false,
                         description{"Generate hits on the fly, not reading from the score"}};

  attribute<symbol> filter_hits{this, "filter_hits", "score_notes",
                         description{"Filter out notes not in the score / use scored velocities"},
                         range{"none", "score_notes", "score_notes_and_vels"}};

  attribute<bool> signal_out{this, "signal_out", true,
                         description{"Output signals"}};

  attribute<bool> message_out{this, "message_out", true,
                         description{"Output messages"}};

  // BOOT STAMP
  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        cout << "rolypoly~ v" << VERSION << " - 2023 Grigore Burloiu - rvirmoors.github.io" << endl;
        at::set_num_threads(1);         // Disables the intraop thread pool.
        at::set_num_interop_threads(1); // Disables the interop thread pool.
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
    if (DEBUG) cout << "== M_TIMER == playhead_ms  | size  :  " << playhead_ms << " | " << score.size(0) << endl;
    if (timer_mode == TIMER::READ) {
      read_deferred.set();
    } else if (timer_mode == TIMER::PLAY) {
      perform_threaded.set();
      if (!done_playing) {
        m_timer.delay(lookahead_ms / 4);
      }
    } else if (timer_mode == TIMER::TRAIN) {
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
        // get any available model outputs from the previous run
        m_compute_thread->join();
      }

      // look for notes up to lookahead_ms
      if (!done_playing)
        advanceReadHead();

      // run the model on notes found
      if (m_use_thread && !done_playing) {
        m_compute_thread = std::make_unique<std::thread>(&rolypoly::tensorToModel, this);
      }
      return {};
    }
  };

  queue<> train_deferred {this, 
    MIN_FUNCTION {
      if (DEBUG) cout << "train_deferred" << endl;
      if (m_train) {
        torch::AutoGradMode enable_grad(true);
//        try {
          backend::TrainConfig config;
          config.lr = 1e-5;
          config.batch_size = 8;
          config.block_size = power_ceil(score_ms.size()/2);
          config.epochs = 10;
          torch::Tensor losses = finetune(config);
          //model->eval();
          cout << "Losses over " << config.epochs << " epochs:\n" << losses << endl;
          cout << "Using epoch w/ smallest loss. To play with this version, send the 'start' message. To save this version, send 'write'. To train again, send 'train'." << endl;
        // }
        // catch (std::exception& e)
        // {
        //     cerr << e.what() << endl;
        // }
        m_train = false;
      }
      return {};
    }
  };

  message<> read {this, "read", "Load score",
    MIN_FUNCTION {
      done_reading = false;
      initialiseScore();
      timer_mode = TIMER::READ;
      m_timer.delay(0);
      return {};
    }
  };

  message<> write {this, "write", "Save finetuned model",
    MIN_FUNCTION {
      if (m_loaded) {
        torch::save(model, "roly_fine.pt");
        cout << "Saved roly_fine.pt" << endl;
      } else {
        cerr << "No model to save!" << endl;
      }
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
      if (m_play || !done_reading) {
        cerr << "wait until the end of the song, then train!" << endl;
        return {};
      }
      
      cout << "Finetuning the model... this could take a while." << endl;
      if (args.size() == 3) {
        train_ops[0] = (double)args[0];
        train_ops[1] = (double)args[1];
        train_ops[2] = (double)args[2];
        cout << "using train options [vel/off/gtr]: " << 
          train_ops[0].item<float>() << " / " <<
          train_ops[1].item<float>() << " / " <<
          train_ops[2].item<float>() << endl;
      } else {
        cout << "using default train options [vel/off/gtr]: " << 
          train_ops[0].item<float>() << " / " <<
          train_ops[1].item<float>() << " / " <<
          train_ops[2].item<float>() << endl;
      }

      m_train = true;
      timer_mode = TIMER::TRAIN;
      m_timer.delay(50);
      
      return {};
    }
  };
};

// ========= ROLYPOLY~ METHOD DEFINITIONS =========

void rolypoly::dumpout(string prefix, double value) {
  m_note[0] = prefix;
  m_note[1] = value;
  m_outlets[INX_DUMPOUT].get()->send(m_note[0], m_note[1]);
}

torch::Tensor rolypoly::finetuneLoss(torch::Tensor out, torch::Tensor x) {
  torch::Tensor vel = x.index({Slice(), Slice(), Slice(0, 9)});
  torch::Tensor vel_hat = out.index({Slice(), Slice(), Slice(0, 9)});

  torch::Tensor tau_g = x.index({Slice(), Slice(), INX_TAU_G});
  torch::Tensor tau_g_hat = out.index({Slice(), Slice(), 18}); // model has 19 outputs, last one is tau_g

  torch::Tensor y_hat_offsets = out.index({Slice(), Slice(), Slice(9, 18)});
  torch::Tensor non_zero_mask = (vel_hat != 0).to(torch::kFloat32);
  torch::Tensor non_zero_sum = (y_hat_offsets * non_zero_mask).sum(2);
  torch::Tensor non_zero_count = non_zero_mask.sum(2);
  torch::Tensor mean_offsets = non_zero_sum / non_zero_count.clamp_min(1);
  torch::Tensor tau_d = torch::where(
      tau_g != 0,
      mean_offsets,
      0.0
  );

  if (DEBUG) cout << "tau out (D):\n" << tau_d[0] << endl << "real tau (G):\n" << tau_g[0] << endl;

  torch::Tensor r = 0.25 * torch::zeros({1}); // TODO offset regularization vs GMD stats
  torch::Tensor v = 0.5  * train_ops[0] * torch::mse_loss(vel_hat, vel);
  torch::Tensor o = 0.02 * train_ops[1] * torch::mse_loss(tau_d, tau_g);
  torch::Tensor g = 0.02 * train_ops[2] * torch::mse_loss(tau_g_hat, tau_g);

  cout << "losses: vel=" << v.item<float>() <<
    " | off=" << o.item<float>() <<
    " | gtr=" << g.item<float>() << endl;

  return r + v + o + g;
}

torch::Tensor rolypoly::finetune(backend::TrainConfig config) {
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(config.lr));
  double min_loss = std::numeric_limits<double>::infinity();
  torch::Tensor loss;
  
  torch::AutoGradMode enable_grad(true);
  model->train();

  // for debugging:
  // for (int i = 0; i < score.size(0); i++) {
  //   if (i % 2) score[i][INX_TAU_G] = -0.02;
  //   else score[i][INX_TAU_G] = 0.04;
  // }

  std::map<std::string, std::vector<torch::Tensor>> train_data;
  train_data["X"].push_back(score.clone().detach());
  train_data["Y"].push_back(score.clone().detach());

  torch::Tensor losses = torch::zeros({0}).to(device);

  for (int epoch = 0; epoch < config.epochs; epoch++) {
    // torch::autograd::DetectAnomalyGuard detect_anomaly;
    optimizer.zero_grad();
        
    torch::Tensor x, y; // y is dummy, unused
    backend::getBatch(train_data, 
        config.batch_size, 
        config.block_size,
        x, y);

    torch::Tensor out = model->forward(x);

    loss = finetuneLoss(out, x);
    if (loss.item<double>() < min_loss) {
      min_loss = loss.item<double>();
      torch::save(model, "roly_best.pt");
      if (DEBUG) cout << "SAVED BEST" << endl;
    }
    loss.backward();
    torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5);
    optimizer.step();

    losses = torch::cat({losses, loss});
  }

  torch::load(model, "roly_best.pt", device);
  return losses;
}

void rolypoly::loadFinetuned(std::string path) {
  torch::load(model, path, device);
  cout << "Loaded finetuned model" << endl;
  // model->eval();
}

void rolypoly::initialiseScore() {
  score = torch::zeros({0, INPUT_DIM}).to(device);
  score_ms.clear();
}

rolypoly::rolypoly(const atoms &args)
    : m_compute_thread(nullptr), m_loaded(false),
      done_reading(false), m_play(false), m_train(false),
      m_use_thread(true), lookahead_ms(500) {

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
    m_path = get_latest_model("roly.pt");
    h_m_path = get_latest_model("roly_hits.pt");

    auto midi_path = std::string(args[0]);
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
      upTo_ms = skip = 0;
  }

  // TRY TO LOAD MODELS
  try {
    torch::load(model, m_path, device);
    torch::load(hitsModel, h_m_path, device);
    // model->eval();
    // hitsModel->eval();
  } catch (std::exception& e)
  {
      if (DEBUG) cerr << e.what() << endl;
      cout << "Error loading models." << endl;
  }
  m_loaded = true;

  // LOAD FINETUNED MODEL IF EXISTS
  try {
    loadFinetuned(get_latest_model("roly_fine.pt"));
  }         
  catch (std::exception& e)
  {
      // if (DEBUG) cerr << e.what() << endl;
      cout << "No finetuned model found." << endl;
  }

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
  m_outlets.push_back(std::make_unique<outlet<>>(
    this, "(list) dump out"));

  cout << "Running warmup, please wait (Max will freeze for a few seconds) ..." << endl;
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
  //if (DEBUG) cout << score.index({Slice(0,5)}) << endl;
  
  if (i >= midifile[1].size()) {
    return true; // done
  }
  return false; // not done yet (SHOULD NEVER HAPPEN)
}

void rolypoly::prepareToPlay() {
  if (m_compute_thread && m_compute_thread->joinable()) {
    m_compute_thread->join();
  }
  playhead_ms = t_toModel = t_fromModel =
    played_ms = t_play = upTo_ms = 0;
  play_notes = torch::zeros({0, INPUT_DIM}).to(device);
  play_notes = torch::cat({play_notes, score[0].unsqueeze(0)}, 0);
  train_ops = torch::tensor({1., 0.4, 0.8}).to(device);
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
  // out: tensor modelOut (1, ...t_toModel, OUTPUT_DIM=19)
  // if in generate mode, then run the HitsModel to insert new note(s) into the score

  int newNotes = t_toModel - t_fromModel;
  torch::Tensor modelOut = torch::zeros({1, 1, OUTPUT_DIM}).to(device);

  if (!newNotes) { // no new notes
    if (DEBUG) cout << "== TENStoMOD == no new notes to compute" << endl;
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
  if (DEBUG) cout << "== TENStoMOD == sending " << start << " - " << t_toModel-1 << endl;
  torch::Tensor input_tensor = score.index({Slice(start, t_toModel)}).unsqueeze(0);
  backend::dataScaleDown(input_tensor);

  // send the notes to the model, to get the offsets for play_notes
  try {
    torch::NoGradGuard no_grad_guard;
    modelOut = model(input_tensor);
    // modelOut = input_tensor.index({Slice(), Slice(0, input_tensor.size(1)), Slice(0, input_tensor.size(2))});
    backend::dataScaleUp(input_tensor);
    backend::dataScaleUp(modelOut);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  // populate play_notes[...t_toModel]
  if (DEBUG) cout << "== TENStoMOD == notes from model == " << t_fromModel << " : " << t_toModel-1 << endl;

  for (int i = modelOut.size(1) - newNotes; i < modelOut.size(1); i++) {
    torch::Tensor new_note;
    // if using score velocities, just take [9:18] (the offsets) from modelOut
    if (static_cast<symbol&>(filter_hits) == "score_notes") {
      auto zero_vels = torch::where( // zero all the vels except the one that's nonzero in the input
        input_tensor[0][i].index({Slice(0, 9)}) == 0,
        0.0,
        modelOut[0][i].index({Slice(0, 9)})
      );
      auto zero_offsets = torch::where( // if vel is zero, zero the offset too
        input_tensor[0][i].index({Slice(0, 9)}) == 0,
        0.0,
        modelOut[0][i].index({Slice(9, 18)})
      );
      new_note = torch::cat({
        zero_vels,
        zero_offsets
          });
    } else if (static_cast<symbol&>(filter_hits) == "score_notes_and_vels") {
      auto zero_offsets = torch::where( // if vel is zero, zero the offset too
        input_tensor[0][i].index({Slice(0, 9)}) == 0,
        0.0,
        modelOut[0][i].index({Slice(9, 18)})
      );
      new_note = torch::cat({
        input_tensor[0][i].index({Slice(0, 9)}),
        zero_offsets
          });
    }
    else // else, take all the velocities+offsets generated by the model
      new_note = modelOut[0][i].index({Slice(0, OUTPUT_DIM - 1)});

    // convert bartime offset values to ms
    new_note.index_put_({Slice(9, 18)},
      bartime_to_ms(
        new_note.index({Slice(9, 18)}),
        input_tensor[0][i][INX_BPM],
        input_tensor[0][i][INX_TSIG]
      )
    );

    new_note = torch::cat({new_note, input_tensor[0][i].index({Slice(INX_BPM, INX_TAU_G)})});
    new_note = torch::cat({new_note, modelOut[0][i][18].unsqueeze(0)}); // last output channel: tau_g_hat
    new_note.unsqueeze_(0);

    play_notes = torch::cat({play_notes, new_note}, 0);
  }
  
  if (generate && playhead_ms > upTo_ms) {
    // compute new notes to be played next = inserted into the score at t_toModel
    // how many? as many as fit in the next lookahead_ms
    upTo_ms = score_ms[t_toModel-1] + lookahead_ms;
    double generated_note_ms = 0.0;
    if (DEBUG) cout << "GENERATE up to " << upTo_ms << ", executing at " << playhead_ms << endl;
    int i = 0;

    while (generated_note_ms < upTo_ms) {
      input_tensor = score.index({Slice(start+i, t_toModel+i)}).clone().detach().unsqueeze(0);
      // zero the tau_g for note generation (hitsModel hasn't seen any tau_g)
      input_tensor.index_put_({Slice(), Slice(), INX_TAU_G}, 0.0);

      if (DEBUG) cout << "pos into hitsModel:\n" << input_tensor.index({Slice(), Slice(), INX_BAR_POS }) << endl;
      backend::dataScaleDown(input_tensor);

      torch::NoGradGuard no_grad_guard;
      hitsModel->train();
      torch::Tensor hitsOut = hitsModel(input_tensor);
      backend::dataScaleUp(input_tensor);
      backend::dataScaleUpHits(hitsOut);
      int last = hitsOut.size(1) - 1;

      if (DEBUG) cout << "pos out from hModel:\n" << hitsOut.index({0, Slice(), 0}).unsqueeze(0) << endl;
      
      torch::Tensor generated_note = torch::cat({
        hitsOut[0][last].index({Slice(1, 10)}), // hits generated
        torch::zeros({9}).to(device), // offsets estimated above, to be filled in below
        score[t_toModel-1].index({Slice(INX_BPM, INX_BAR_POS)}), // tempo & t_sig from prev note
        hitsOut[0][last][0].unsqueeze(0), // bar pos generated
        torch::zeros({1}).to(device), // tau_guitar, to be filled in on onset detect
      });

      score = torch::cat({
        score.index({Slice(0, t_toModel+i)}),
        generated_note.unsqueeze(0),
        score.index({Slice(t_toModel+i, None)})
          }, 0);

      if (DEBUG) cout << "testing " << generated_note[INX_BAR_POS].item<double>() << endl;

      // if the generated bar_pos is before the preceding note's bar_pos
      // note: JANKY!!!! TODO find something better...
      if (generated_note[INX_BAR_POS].item<double>() < 
        score[t_toModel+i - 1][INX_BAR_POS].item<double>()) {
        // heuristic to check end of bar
        // (and avoid stuff like 0.8 0.2 0.05 where the 0.2 should be 0.9)
        input_tensor = score.index({Slice(start+1+i, t_toModel+i)}).unsqueeze(0);
        backend::dataScaleDown(input_tensor);
        torch::Tensor future_note = hitsModel(input_tensor);
        backend::dataScaleUp(input_tensor);
        last = future_note.size(1) - 1;
        double future_bar_pos = future_note[0][last][0].item<double>();
        if (DEBUG) cout << "future would be @ " << future_bar_pos << "<>" << generated_note[INX_BAR_POS].item<double>() << endl;
        if (future_bar_pos < generated_note[INX_BAR_POS].item<double>()) {
          // case: 0.8 0.5 0.2 where 0.5 should be 0.9
          // move the generated note bar_pos back to the end of the bar
          generated_note[INX_BAR_POS] = (score.index({t_toModel+i-1, INX_BAR_POS}) + 1.) / 2.;
          score[t_toModel+i][INX_BAR_POS] = generated_note[INX_BAR_POS];
          if (DEBUG) cout << "adjusted pos out: " << generated_note[INX_BAR_POS].item<double>() << endl;
        }
        else if (future_bar_pos > score.index({t_toModel+i-1, INX_BAR_POS}).item<double>()) {
          // case: 0.8 0.75 0.9 where 0.75 should be 0.9
          // discard the generated note (0.75) and use the future note (0.9) instead
          generated_note = torch::cat({
            future_note[0][last].index({Slice(1, 10)}), // hits generated
            torch::zeros({9}).to(device), // offsets
            score[t_toModel-1].index({Slice(INX_BPM, INX_BAR_POS)}), // tempo & t_sig from prev note
            future_note[0][last][0].unsqueeze(0), // bar pos generated
            torch::zeros({1}).to(device), // tau_guitar, to be filled in on onset detect
          });
          score[t_toModel+i] = generated_note;
          if (DEBUG) cout << "replaced pos out: " << generated_note[INX_BAR_POS].item<double>() << endl;
        }
      }

      // now compute the ms equivalent of the bar_pos
      generated_note_ms = score_ms[t_toModel+i-1];
      if (DEBUG) cout << "push_ms = b_to_ms(" << score[t_toModel+i - 1][INX_BAR_POS].item<double>() << ", " << score[t_toModel+i][INX_BAR_POS].item<double>() << ", " << score[t_toModel+i][INX_BPM].item<double>() << ", " << score[t_toModel+i][INX_TSIG].item<double>() << endl;
      double push_ms = bartime_to_ms(score[t_toModel+i - 1][INX_BAR_POS].item<double>(), 
                          score[t_toModel+i][INX_BAR_POS].item<double>(), 
                          score[t_toModel+i][INX_BPM].item<double>(),
                          score[t_toModel+i][INX_TSIG].item<double>());

      if (DEBUG) cout << "insert note at " << generated_note_ms << " + " << push_ms << endl;
      score_ms.insert(score_ms.begin() + t_toModel+i, generated_note_ms);

      // delay all notes starting from the generated_note, by push_ms
      std::transform(score_ms.begin() + t_toModel+i, score_ms.end(), score_ms.begin() + t_toModel+i,
        [&push_ms](auto t) {return t + push_ms;} );
      generated_note_ms += push_ms; // for the while() check above

      if (DEBUG) cout << "BEF PUSHBAR " << score.index({Slice(), INX_BAR_POS}).unsqueeze(0) << endl;
      double push_bar = generated_note[INX_BAR_POS].item<double>() - score[t_toModel+i - 1][INX_BAR_POS].item<double>() + 1.0;
      score.index_put_({Slice(t_toModel+i+1, None), INX_BAR_POS},
        (score.index({Slice(t_toModel+i+1, None), INX_BAR_POS}) + push_bar).frac_()
        );
      if (DEBUG) cout << "AFT PUSHBAR " << score.index({Slice(), INX_BAR_POS}).unsqueeze(0) << endl;

      if (DEBUG) cout << score_ms << endl;
      if (DEBUG) cout << "Score size increased to: " << score.size(0) << endl;

      assert(score.size(0) == score_ms.size());
      i++;
    }
    if (DEBUG) cout << "GENERATED " << i << " notes. Finished executing at " << playhead_ms << endl;
    // m_play = false;
    // done_playing = true;
    // timer_mode = TIMER::INACTIVE;
    // m_timer.stop();
  }

  // copy executed offsets to score (to be later fed into the model for inference)
  if (t_toModel < score.size(0) - 1) {
    auto zero_offsets = torch::where( // if vel is zero, zero the offset too
      play_notes.index({Slice(t_fromModel, t_toModel), Slice(0, 9)}) == 0,
      0.0,
      play_notes.index({Slice(t_fromModel, t_toModel), Slice(9, 18)})
    );
    score.index_put_({Slice(t_fromModel, t_toModel), Slice(9, 18)},
      ms_to_bartime(
        zero_offsets,
        score.index({Slice(t_fromModel, t_toModel), Slice(INX_BPM, INX_BPM+1)}),
        score.index({Slice(t_fromModel, t_toModel), Slice(INX_TSIG, INX_TSIG+1)})
      )
    );
    if(static_cast<symbol&>(filter_hits) == "none") { // add played velocities
      score.index_put_({Slice(t_fromModel, t_toModel), Slice(0, 9)},
        play_notes.index({Slice(t_fromModel, t_toModel), Slice(0, 9)})
      );
    }
  }
  t_fromModel = t_toModel;
}

std::pair<double, int> rolypoly::computeNextNoteTimeMs() {
  if (!done_playing) {
    if (t_play >= play_notes.size(0)) {
      if (DEBUG) cout << "no tau in play_notes yet: " << t_play << " >= " << play_notes.size(0) << endl;
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
      return std::make_pair(earliest_ms, earliest_channel - 9);
    }
    else {
      t_play++;
      if (DEBUG) cout << "next note: " << t_play << endl;
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

  try {  // get the onset time in ms
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
    if (DEBUG) cout << "NOT within " << closest_note_duration/3 << " of " << closest_note_time << endl; 
    return;
  }

  // add detected tau_g to score
  score[closest_note][INX_TAU_G] = ms_to_bartime(tau_guitar, score[closest_note][INX_BPM], score[closest_note][INX_TSIG]);}
  catch (const std::exception& e) {
    std::cerr << "ONSET ERROR:" << e.what() << std::endl;
  }
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

    // dumpout
    dumpout("playhead_ms", playhead_ms);
    dumpout("upTo_ms", upTo_ms);

    if (!play_notes.size(0)) {
      // no notes yet!
      return;
    }

    if (t_play == 0) { // play the first note
      for (int i = 0; i < 9; i++) {
        auto out = output.samples(i);
        double vel = score[0][i].item<double>();
        if (vel > 0.1) {
          if (signal_out) // OUTPUT NOTE SIGNALS
            out[0] = std::max(std::min(vel / 127., 1.), 0.1);
          if (message_out) { // OUTPUT MESSAGES
            int msg_index = output.channel_count();
            m_note[0] = out_pitches[i];
            m_note[1] = vel;
            m_outlets[msg_index].get()->send(m_note[0], m_note[1]);
          }
        }
      }
      t_play++;
      return;
    }

    // increment playhead
    double bufsize_ms = lib::math::samples_to_milliseconds(vec_size, samplerate());
    std::pair<double, int> next_note = computeNextNoteTimeMs(); // usually it's the next note that doesn't have a tau yet

    // but at the end of the score, it can be the final note (computeNext...)
    next_note.first = std::min(score_ms[t_toModel], next_note.first );
    if (playhead_ms < next_note.first)
      playhead_ms += bufsize_ms;

    // if (DEBUG) cout << "playhead: " << playhead_ms << " | next note @ " << next_note.first << endl;
    while (playhead_ms >= next_note.first - bufsize_ms && !done_playing) {
      if (next_note.second == -1) {
        // no tau yet
        return;
      }
      // when the time comes, play the microtime-adjusted note
      long micro_index = (next_note.first - playhead_ms) / bufsize_ms * vec_size;
      micro_index = std::min(micro_index, vec_size-1);
      micro_index = std::max(micro_index, 0L);
      auto out = output.samples(next_note.second);
      double vel = play_notes[t_play][next_note.second].item<double>();
      if (vel > 0.1) {
        if (signal_out) // OUTPUT NOTE SIGNALS
          out[micro_index] = std::max(std::min(vel / 127., 1.), 0.1);
        if (message_out) { // OUTPUT MESSAGES
          int msg_index = output.channel_count();
          m_note[0] = out_pitches[next_note.second];
          m_note[1] = vel;
          m_outlets[msg_index].get()->send(m_note[0], m_note[1]);
        }
      }
      played_ms = next_note.first;
      next_note = computeNextNoteTimeMs();
      //next_note.first = std::max(score_ms[t_toModel], next_note.first );
    }

    if (playhead_ms >= score_ms.back() || t_play >= score.size(0) - 1) {
      cout << "Done playing. To finetune the model based on this run, send the 'train' message." << endl;
      m_play = false;
      done_playing = true;
      timer_mode = TIMER::INACTIVE;
      m_timer.stop();
      if (m_compute_thread && m_compute_thread->joinable()) {
        m_compute_thread->join();
        if (DEBUG) cout << "== END == JOINED THREAD" << endl;
      }
    }
  } else {
    fill_with_zero(output);
  }
};

MIN_EXTERNAL(rolypoly);