#include <iostream>
#include <string>
#include "feature_extract.h"
#include "predictor.h"
#include "visualizer.h"

int main() {
    std::string imagePath;
    std::string viewType;
    std::cout << "=== Rhinoplasty AI ===" << std::endl;
    std::cout << "Enter path to facial image: ";
    std::getline(std::cin, imagePath);

    std::cout << "Select view type (front/side): ";
    std::getline(std::cin, viewType);

    // Step 1: Load and extract features
    std::vector<float> features = extractNoseFeatures(imagePath, viewType);
    if (features.empty()) {
        std::cerr << "Failed to extract features from the image." << std::endl;
        return 1;
    }

    // Step 2: Predict new nose scale/factor
    float predictedScale = predictNoseModification(features);
    std::cout << "Predicted nose scale factor: " << predictedScale << std::endl;

    // Step 3: Apply visual modification
    std::string outputPath = "output/modified_image.png";
    bool success = visualizeNoseModification(imagePath, viewType, predictedScale, outputPath);

    if (success) {
        std::cout << "Modified image saved to: " << outputPath << std::endl;
    } else {
        std::cerr << "Failed to apply visual modification." << std::endl;
        return 1;
    }

    std::cout << "=== Process Complete ===" << std::endl;
    return 0;
}
