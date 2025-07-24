#pragma once

#include <string>
#include <memory>
#include <vector>
#include <tensorflow/c/c_api.h>

class SpleeterProcessor
{
public:
  // Constants
  static constexpr int MODEL_SAMPLE_RATE = 44100;
  static constexpr int MODEL_CHANNELS = 2;

  struct AudioData
  {
    std::vector<float> samples;
    int sampleRate;
    int channels;
    int sampleCount;
  };

  // Constructor and destructor
  SpleeterProcessor();
  ~SpleeterProcessor();

  // Main processing function, call it as you like.
  bool process(const std::string &inputPath, const std::string &outputVocalsPath, int fileDuration, int chunkSeconds);

  // Model initialization, only call it once.
  bool initializeModel();

private:
  struct TFResources
  {
    TF_Status *status = nullptr;
    TF_Graph *graph = nullptr;
    TF_Session *session = nullptr;

    // Constructor
    TFResources() = default;

    // Destructor - automatically cleans up
    ~TFResources()
    {
      if (session)
      {
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
        session = nullptr;
      }
      if (graph)
      {
        TF_DeleteGraph(graph);
        graph = nullptr;
      }
      if (status)
      {
        TF_DeleteStatus(status);
        status = nullptr;
      }
    }

    // Prevent copying
    TFResources(const TFResources &) = delete;
    TFResources &operator=(const TFResources &) = delete;
  };

  TFResources resources_;

  std::string getModelPath() const;
  bool loadModel(const std::string &modelPath);
  bool concatenateChunks(const std::string &outputBasePath, size_t totalChunks);
  bool savePCMFile(const std::string &path, const float *data, size_t sampleCount) const;
  bool runInference(const AudioData &input, float **vocalsOut);
  void cleanup();

  // Prevent copying
  SpleeterProcessor(const SpleeterProcessor &) = delete;
  SpleeterProcessor &operator=(const SpleeterProcessor &) = delete;
};