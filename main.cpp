#include "SpleeterProcessor.hpp"
#include <iostream>
#include <filesystem>

int main(int argc, char *argv[])
{

    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <input_pcms_path> <output_pcms_path> <file_duration> <chunk_seconds>" << std::endl;
        return 1;
    }

    try
    {
        auto processor = std::make_unique<SpleeterProcessor>();

        if (!processor->initializeModel())
        {
            std::cerr << "Failed to initialize model" << std::endl;
            return 1;
        }
        const int fileDuration = std::stoi(argv[3]);
        const int chunkSeconds = std::stoi(argv[4]);

        if (!processor->process(argv[1], argv[2], fileDuration, chunkSeconds))
        {
            std::cerr << "Processing failed" << std::endl;
            return 1;
        }
        std::cout << "Processing completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}