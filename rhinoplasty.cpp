#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <xgboost/c_api.h>

namespace fs = std::filesystem;

// Global model handle
BoosterHandle booster;

// Function to extract HOG features
std::vector<float> extractHOG(const cv::Mat& img) {
    cv::HOGDescriptor hog;
    std::vector<float> descriptors;
    hog.compute(img, descriptors);
    return descriptors;
}

// Load images and extract features
void extractFeaturesFromFolder(const std::string& folderPath, std::vector<std::vector<float>>& features) {
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            cv::Mat img = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
            if (!img.empty()) {
                cv::resize(img, img, cv::Size(128, 128));
                auto hog_features = extractHOG(img);
                features.push_back(hog_features);
            }
        }
    }
}

// Save extracted features to CSV
void saveFeaturesToCSV(const std::string& filename, const std::vector<std::vector<float>>& features, const std::vector<float>& labels) {
    std::ofstream file(filename);
    for (size_t i = 0; i < features.size(); ++i) {
        for (float val : features[i]) {
            file << val << ",";
        }
        file << labels[i] << "\n";
    }
    file.close();
}

// Load CSV to feature matrix and label vector
void loadCSV(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& labels) {
    std::ifstream file(filename);
    std::string line;
    while (getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string val;
        while (getline(ss, val, ',')) {
            row.push_back(std::stof(val));
        }
        labels.push_back(row.back());
        row.pop_back();
        features.push_back(row);
    }
    file.close();
}

// Train XGBoost model
void trainModel(const std::vector<std::vector<float>>& X, const std::vector<float>& y) {
    DMatrixHandle dtrain;
    std::vector<float> flat_X;
    for (const auto& row : X)
        flat_X.insert(flat_X.end(), row.begin(), row.end());

    XGDMatrixCreateFromMat(flat_X.data(), X.size(), X[0].size(), -1, &dtrain);
    XGDMatrixSetFloatInfo(dtrain, "label", y.data(), y.size());

    std::vector<const char*> param = { "max_depth", "3", "eta", "0.1", "objective", "reg:squarederror" };
    XGBoosterCreate(&dtrain, 1, &booster);
    XGBoosterSetParam(booster, "eval_metric", "rmse");
    XGBoosterSetParam(booster, "verbosity", "1");

    for (int i = 0; i < 100; ++i)
        XGBoosterUpdateOneIter(booster, i, dtrain);

    XGBoosterSaveModel(booster, "model.xgb");
    XGDMatrixFree(dtrain);

    std::cout << "Model saved as model.xgb" << std::endl;
}

// Predict on test set
void predict(const std::vector<std::vector<float>>& X_test, std::vector<float>& predictions) {
    DMatrixHandle dtest;
    std::vector<float> flat_X;
    for (const auto& row : X_test)
        flat_X.insert(flat_X.end(), row.begin(), row.end());

    XGDMatrixCreateFromMat(flat_X.data(), X_test.size(), X_test[0].size(), -1, &dtest);

    bst_ulong out_len;
    const float* out_result;
    XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result);

    predictions.assign(out_result, out_result + out_len);
    XGDMatrixFree(dtest);
}

int main() {
    std::vector<std::vector<float>> features;
    std::vector<float> labels;

    // 1. Extract features from folder
    std::string dataset_path = "A:/projects/Rhinoplasty_ai/Humans"; // Change as needed
    extractFeaturesFromFolder(dataset_path, features);

    // 2. Dummy labels (replace with real target values if available)
    labels.resize(features.size(), 1.0f); // Example: all 1.0

    // 3. Save features to CSV
    saveFeaturesToCSV("features.csv", features, labels);

    // 4. Load features for training
    std::vector<std::vector<float>> X;
    std::vector<float> y;
    loadCSV("features.csv", X, y);

    // 5. Train model
    trainModel(X, y);

    // 6. Predict using same features (for test)
    std::vector<float> preds;
    predict(X, preds);

    for (size_t i = 0; i < preds.size() && i < 5; ++i)
        std::cout << "Prediction " << i + 1 << ": " << preds[i] << std::endl;

    XGBoosterFree(booster);
    return 0;
}
