#pragma once
#include "matrix.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <iostream>
#include <numeric>
#include <stdexcept>

class Dataset {
private:
    Matrix X_;               // Features
    Matrix Y_;               // Labels
    Matrix mean_;            // Mean (for std normalization)
    Matrix std_;             // Std (for std normalization)
    Matrix min_;             // Min (for min-max normalization)
    Matrix max_;             // Max (for min-max normalization)

    enum class NormType { NONE, STANDARD, MINMAX };
    NormType norm_type_ = NormType::NONE;

public:
    Dataset() = default;

    // ======================
    // LOAD FROM CSV
    // ======================
    bool load_csv(const std::string& filename, bool has_header = true, bool last_column_is_label = true) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }

        std::vector<std::vector<double>> features;
        std::vector<std::vector<double>> labels;
        std::string line;

        if (has_header)
            std::getline(file, line); // skip header

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row_values;

            while (std::getline(ss, cell, ','))
                row_values.push_back(std::stod(cell));

            if (last_column_is_label) {
                labels.push_back({ row_values.back() });
                row_values.pop_back();
            }
            features.push_back(row_values);
        }

        size_t rows = features.size();
        size_t cols = features[0].size();
        X_ = Matrix(rows, cols);
        Y_ = Matrix(rows, 1);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j)
                X_(i, j) = features[i][j];
            if (last_column_is_label)
                Y_(i, 0) = labels[i][0];
        }

        file.close();
        norm_type_ = NormType::NONE;
        return true;
    }

    // ======================
    // GETTERS
    // ======================
    const Matrix& X() const { return X_; }
    const Matrix& Y() const { return Y_; }
    size_t rows() const { return X_.rows(); }
    size_t cols() const { return X_.cols(); }

    // ======================
    // STANDARD NORMALIZATION (Z-SCORE)
    // ======================
    void normalize_standard() {
        size_t cols = X_.cols(), rows = X_.rows();
        mean_ = Matrix(1, cols);
        std_ = Matrix(1, cols);

        for (size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < rows; ++i) sum += X_(i, j);
            double mean = sum / rows;
            mean_(0, j) = mean;

            double sq_sum = 0.0;
            for (size_t i = 0; i < rows; ++i)
                sq_sum += (X_(i, j) - mean) * (X_(i, j) - mean);
            double stdev = std::sqrt(sq_sum / rows);
            std_(0, j) = stdev == 0.0 ? 1.0 : stdev;

            for (size_t i = 0; i < rows; ++i)
                X_(i, j) = (X_(i, j) - mean) / stdev;
        }

        norm_type_ = NormType::STANDARD;
    }

    Matrix denormalize_standard(const Matrix& normalized_X) const {
        if (norm_type_ != NormType::STANDARD)
            throw std::runtime_error("Dataset not standard-normalized.");

        Matrix result = normalized_X;
        for (size_t j = 0; j < result.cols(); ++j)
            for (size_t i = 0; i < result.rows(); ++i)
                result(i, j) = result(i, j) * std_(0, j) + mean_(0, j);
        return result;
    }

    // ======================
    // MIN-MAX NORMALIZATION
    // ======================
    void normalize_minmax(double new_min = 0.0, double new_max = 1.0) {
        size_t cols = X_.cols(), rows = X_.rows();
        min_ = Matrix(1, cols);
        max_ = Matrix(1, cols);

        for (size_t j = 0; j < cols; ++j) {
            double min_val = X_(0, j), max_val = X_(0, j);
            for (size_t i = 1; i < rows; ++i) {
                if (X_(i, j) < min_val) min_val = X_(i, j);
                if (X_(i, j) > max_val) max_val = X_(i, j);
            }
            min_(0, j) = min_val;
            max_(0, j) = max_val;

            double range = (max_val - min_val == 0) ? 1.0 : (max_val - min_val);
            for (size_t i = 0; i < rows; ++i)
                X_(i, j) = ((X_(i, j) - min_val) / range) * (new_max - new_min) + new_min;
        }

        norm_type_ = NormType::MINMAX;
    }

    Matrix denormalize_minmax(const Matrix& normalized_X, double orig_min = 0.0, double orig_max = 1.0) const {
        if (norm_type_ != NormType::MINMAX)
            throw std::runtime_error("Dataset not min-max normalized.");

        Matrix result = normalized_X;
        for (size_t j = 0; j < result.cols(); ++j) {
            double min_val = min_(0, j);
            double max_val = max_(0, j);
            double range = (max_val - min_val == 0) ? 1.0 : (max_val - min_val);
            for (size_t i = 0; i < result.rows(); ++i)
                result(i, j) = ((result(i, j) - orig_min) / (orig_max - orig_min)) * range + min_val;
        }
        return result;
    }

    // ======================
    // TRAIN / TEST SPLIT
    // ======================
    std::pair<Dataset, Dataset> train_test_split(double test_ratio = 0.2, unsigned seed = 42) const {
        if (X_.rows() != Y_.rows())
            throw std::runtime_error("Feature and label size mismatch.");

        size_t n_samples = X_.rows();
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);

        size_t test_size = static_cast<size_t>(n_samples * test_ratio);
        size_t train_size = n_samples - test_size;

        Dataset train_set, test_set;
        train_set.X_ = Matrix(train_size, X_.cols());
        train_set.Y_ = Matrix(train_size, Y_.cols());
        test_set.X_ = Matrix(test_size, X_.cols());
        test_set.Y_ = Matrix(test_size, Y_.cols());

        for (size_t i = 0; i < train_size; ++i) {
            size_t idx = indices[i];
            for (size_t j = 0; j < X_.cols(); ++j)
                train_set.X_(i, j) = X_(idx, j);
            train_set.Y_(i, 0) = Y_(idx, 0);
        }

        for (size_t i = 0; i < test_size; ++i) {
            size_t idx = indices[train_size + i];
            for (size_t j = 0; j < X_.cols(); ++j)
                test_set.X_(i, j) = X_(idx, j);
            test_set.Y_(i, 0) = Y_(idx, 0);
        }

        return { train_set, test_set };
    }
};
