#include "SpleeterProcessor.hpp"
#include <iostream>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

SpleeterProcessor::SpleeterProcessor() = default;

SpleeterProcessor::~SpleeterProcessor()
{
  cleanup();
}

bool SpleeterProcessor::initializeModel()
{
  try
  {
    std::string modelPath = getModelPath();
    return loadModel(modelPath);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Initialization error: " << e.what() << std::endl;
    return false;
  }
}

std::string SpleeterProcessor::getModelPath() const
{
    fs::path currentPath = fs::current_path() / "models" / "2stems";
    
    if (!fs::exists(currentPath)) {
      currentPath = "/usr/lib/Basset/models/2stems";
      if (!fs::exists(currentPath)) {
        throw std::runtime_error("Model folder not found at: " + currentPath.string());
      }
    }
    
    return currentPath.string();
}

bool SpleeterProcessor::loadModel(const std::string &modelPath)
{
  resources_.status = TF_NewStatus();
  resources_.graph = TF_NewGraph();

  auto sessionOpts = TF_NewSessionOptions();
  auto runOpts = TF_NewBuffer();
  auto metaGraphDef = TF_NewBuffer();

  const char *tags[] = {"serve"};

  resources_.session = TF_LoadSessionFromSavedModel(
      sessionOpts, runOpts,
      modelPath.c_str(),
      tags, 1,
      resources_.graph,
      metaGraphDef,
      resources_.status);

  TF_DeleteSessionOptions(sessionOpts);
  TF_DeleteBuffer(runOpts);
  TF_DeleteBuffer(metaGraphDef);

  if (TF_GetCode(resources_.status) != TF_OK)
  {
    std::cerr << "Failed to load model: " << TF_Message(resources_.status) << std::endl;
    return false;
  }

  return true;
}

bool SpleeterProcessor::savePCMFile(const std::string &path, const float *data, size_t sampleCount) const
{
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open())
  {
    std::cerr << "Failed to open output file: " << path << std::endl;
    return false;
  }

  file.write(reinterpret_cast<const char *>(data),
             sampleCount * MODEL_CHANNELS * sizeof(float));
  return true;
}

bool SpleeterProcessor::runInference(const AudioData &input, float **vocalsOut)
{
  // Setup input tensor
  int64_t inputDims[] = {input.sampleCount, MODEL_CHANNELS};
  auto inputTensor = TF_NewTensor(
      TF_FLOAT,
      inputDims, 2,
      const_cast<float *>(input.samples.data()), // Remove const with const_cast
      input.samples.size() * sizeof(float),
      [](void *, size_t, void *) {}, nullptr);

  // Setup operations - only for vocals
  TF_Output input_op = {TF_GraphOperationByName(resources_.graph, "input_waveform"), 0};
  TF_Output output_op = {TF_GraphOperationByName(resources_.graph, "output_vocals"), 0};
  TF_Tensor *outputTensor = nullptr;

  // Run inference
  TF_SessionRun(
      resources_.session, nullptr,
      &input_op, &inputTensor, 1,
      &output_op, &outputTensor, 1,
      nullptr, 0, nullptr, resources_.status);

  bool success = (TF_GetCode(resources_.status) == TF_OK);
  if (success)
  {
    size_t dataSize = input.sampleCount * MODEL_CHANNELS * sizeof(float);
    *vocalsOut = new float[input.sampleCount * MODEL_CHANNELS];
    memcpy(*vocalsOut, TF_TensorData(outputTensor), dataSize);
  }

  // Cleanup
  TF_DeleteTensor(inputTensor);
  TF_DeleteTensor(outputTensor);

  return success;
}

bool SpleeterProcessor::concatenateChunks(const std::string &outputBasePath, size_t totalChunks)
{
  std::string finalOutputPath = outputBasePath + ".pcm";
  std::ofstream outFile(finalOutputPath, std::ios::binary);
  if (!outFile.is_open())
  {
    std::cerr << "Failed to create final output file" << std::endl;
    return false;
  }

  for (size_t i = 0; i < totalChunks; i++)
  {
    std::string chunkPath = outputBasePath + "_" + std::to_string(i) + ".pcm";
    std::ifstream chunkFile(chunkPath, std::ios::binary);

    if (!chunkFile.is_open())
    {
      std::cerr << "Failed to open chunk: " << chunkPath << std::endl;
      return false;
    }

    outFile << chunkFile.rdbuf();
    chunkFile.close();

    // Optionally delete the chunk file
    std::filesystem::remove(chunkPath);
  }

  outFile.close();
  return true;
}

bool SpleeterProcessor::process(const std::string &inputPath, const std::string &outputBasePath, int fileDuration, int chunkSeconds)
{
  try
  {
    std::ifstream inputFile(inputPath, std::ios::binary);
    if (!inputFile.is_open())
    {
      throw std::runtime_error("Failed to open input file: " + inputPath);
    }

    const size_t samplesPerChunk = MODEL_SAMPLE_RATE * chunkSeconds;
    const size_t bytesPerChunk = samplesPerChunk * MODEL_CHANNELS * sizeof(float);
    const size_t totalChunks = (fileDuration + chunkSeconds - 1) / chunkSeconds;

    // Use RAII for chunk buffe
    std::vector<float> chunkBuffer(samplesPerChunk * MODEL_CHANNELS);

    for (size_t i = 0; i < totalChunks; i++)
    {
      // Clear buffer before each read
      chunkBuffer.clear();
      chunkBuffer.resize(samplesPerChunk * MODEL_CHANNELS);

      // Read chunk
      inputFile.read(reinterpret_cast<char *>(chunkBuffer.data()), bytesPerChunk);
      size_t samplesRead = inputFile.gcount() / (MODEL_CHANNELS * sizeof(float));

      if (samplesRead == 0)
        break;

      // Process chunk
      AudioData chunkAudio;
      chunkAudio.samples = std::move(chunkBuffer); // Move ownership
      chunkAudio.sampleCount = static_cast<int>(samplesRead);
      chunkAudio.sampleRate = MODEL_SAMPLE_RATE;
      chunkAudio.channels = MODEL_CHANNELS;

      float *rawVocalsData = nullptr;
      bool inferenceSuccess = false;

      try
      {
        inferenceSuccess = runInference(chunkAudio, &rawVocalsData);
        if (!inferenceSuccess)
        {
          delete[] rawVocalsData; // Cleanup if inference failed
          return false;
        }
      }
      catch (...)
      {
        delete[] rawVocalsData; // Cleanup on exception
        throw;
      }

      // Use RAII for vocals data
      std::unique_ptr<float[]> vocalsData(rawVocalsData);

      // Generate output filename for this chunk
      std::string outputPath = outputBasePath + "_" + std::to_string(i) + ".pcm";

      // Save chunk
      if (!savePCMFile(outputPath, vocalsData.get(), samplesRead))
      {
        return false;
      }

      // Calculate and show progress
      float progress = (static_cast<float>(i) / totalChunks) * 100.0f;

      std::cout << "Progress: " << progress << "%" << std::endl;

      // Return buffer for next iteration
      chunkBuffer = std::move(chunkAudio.samples);
    }

    inputFile.close();

    if (!concatenateChunks(outputBasePath, totalChunks))
    {
      std::cerr << "Failed to concatenate chunks" << std::endl;
      return false;
    }

    return true;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Processing error: " << e.what() << std::endl;
    return false;
  }
}

void SpleeterProcessor::cleanup()
{
  if (resources_.session)
  {
    TF_CloseSession(resources_.session, resources_.status);
    TF_DeleteSession(resources_.session, resources_.status);
    resources_.session = nullptr;
  }

  if (resources_.graph)
  {
    TF_DeleteGraph(resources_.graph);
    resources_.graph = nullptr;
  }

  if (resources_.status)
  {
    TF_DeleteStatus(resources_.status);
    resources_.status = nullptr;
  }
}