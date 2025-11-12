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

    Matrix mean_X_;
    Matrix std_X_;
    Matrix min_X_;
    Matrix max_X_;

    Matrix mean_Y_;
    Matrix std_Y_;
    Matrix min_Y_;
    Matrix max_Y_;

    std::vector<std::string> feature_names_;
    std::vector<std::string> target_names_;

    enum class NormType { NONE, STANDARD, MINMAX };
    NormType norm_type_ = NormType::NONE;

public:
    Dataset() = default;
    Dataset(Matrix& X, Matrix& Y);

    // Getters
    const Matrix& X() const;
    const Matrix& Y() const;
    size_t rows() const;
    size_t cols() const;
    const std::vector<std::string>& feature_names() const;
    const std::vector<std::string>& target_names() const;

    // Normalizations
    void normalize_standard_self(bool normalize_targets);
    Dataset normalize_standard(bool normalize_targets);
    void normalize_minmax_self(double new_min, double new_max, bool normalize_targets);
    Dataset normalize_minmax(double new_min, double new_max, bool normalize_targets);

    // Denormalizations
    Matrix denormalize_standard_X(const Matrix& normalized_X) const;
    Matrix denormalize_standard_Y(const Matrix& normalized_Y) const;
    Matrix denormalize_minmax_X(const Matrix& normalized_X, double orig_min, double orig_max) const;
    Matrix denormalize_minmax_Y(const Matrix& normalized_Y, double orig_min, double orig_max) const;

    // Utilities
    bool load_csv(const std::string& filename, bool has_header, size_t num_target);
    Dataset shuffle(size_t seed) const;
    void shuffle_self(size_t seed);
    std::pair<Dataset, Dataset> train_test_split(double train_ratio, unsigned seed) const;

};

// Constructors
inline Dataset::Dataset(Matrix& X, Matrix& Y) : X_(X), Y_(Y)
{
}

// GETTERS
const Matrix& Dataset::X() const { return X_; }
const Matrix& Dataset::Y() const { return Y_; }
size_t Dataset::rows() const { return X_.rows(); }
size_t Dataset::cols() const { return X_.cols(); }
const std::vector<std::string>& Dataset::feature_names() const { return feature_names_; }
const std::vector<std::string>& Dataset::target_names() const { return target_names_; }


// STANDARD NORMALIZATION (Z-SCORE)
inline void Dataset::normalize_standard_self(bool normalize_targets = false) {
    size_t rows = X_.rows(), feat_cols = X_.cols();
    mean_X_ = Matrix(1, feat_cols);
    std_X_ = Matrix(1, feat_cols);

    // --- Features ---
    for (size_t j = 0; j < feat_cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < rows; ++i) sum += X_(i, j);
        double mean = sum / rows;
        mean_X_(0, j) = mean;

        double sq_sum = 0.0;
        for (size_t i = 0; i < rows; ++i)
            sq_sum += (X_(i, j) - mean) * (X_(i, j) - mean);
        double s = std::sqrt(sq_sum / rows);
        std_X_(0, j) = s == 0.0 ? 1.0 : s;

        for (size_t i = 0; i < rows; ++i)
            X_(i, j) = (X_(i, j) - mean) / std_X_(0, j);
    }

    // --- Targets (optional) ---
    if (normalize_targets) {
        size_t t_cols = Y_.cols();
        mean_Y_ = Matrix(1, t_cols);
        std_Y_ = Matrix(1, t_cols);

        for (size_t j = 0; j < t_cols; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < rows; ++i) sum += Y_(i, j);
            double mean = sum / rows;
            mean_Y_(0, j) = mean;

            double sq_sum = 0.0;
            for (size_t i = 0; i < rows; ++i)
                sq_sum += (Y_(i, j) - mean) * (Y_(i, j) - mean);
            double s = std::sqrt(sq_sum / rows);
            std_Y_(0, j) = s == 0.0 ? 1.0 : s;

            for (size_t i = 0; i < rows; ++i)
                Y_(i, j) = (Y_(i, j) - mean) / std_Y_(0, j);
        }
    }

    norm_type_ = NormType::STANDARD;
}

inline Dataset Dataset::normalize_standard(bool normalize_targets = false) {
    Dataset temp(*this);
    temp.normalize_standard_self(normalize_targets);
    return temp;
}

// DENORMALIZE
inline Matrix Dataset::denormalize_standard_X(const Matrix& normalized_X) const {
    Matrix result = normalized_X;
    for (size_t j = 0; j < result.cols(); ++j)
        for (size_t i = 0; i < result.rows(); ++i)
            result(i, j) = result(i, j) * std_X_(0, j) + mean_X_(0, j);
    return result;
}

inline Matrix Dataset::denormalize_standard_Y(const Matrix& normalized_Y) const {
    Matrix result = normalized_Y;
    for (size_t j = 0; j < result.cols(); ++j)
        for (size_t i = 0; i < result.rows(); ++i)
            result(i, j) = result(i, j) * std_Y_(0, j) + mean_Y_(0, j);
    return result;
}



// MIN-MAX NORMALIZATION
inline void Dataset::normalize_minmax_self(double new_min = 0.0, double new_max = 1.0, bool normalize_targets = false) {
    size_t rows = X_.rows(), feat_cols = X_.cols();
    min_X_ = Matrix(1, feat_cols);
    max_X_ = Matrix(1, feat_cols);

    // --- Features ---
    for (size_t j = 0; j < feat_cols; ++j) {
        double min_val = X_(0, j), max_val = X_(0, j);
        for (size_t i = 1; i < rows; ++i) {
            min_val = std::min(min_val, X_(i, j));
            max_val = std::max(max_val, X_(i, j));
        }
        min_X_(0, j) = min_val;
        max_X_(0, j) = max_val;
        double range = (max_val - min_val == 0.0) ? 1.0 : (max_val - min_val);

        for (size_t i = 0; i < rows; ++i)
            X_(i, j) = ((X_(i, j) - min_val) / range) * (new_max - new_min) + new_min;
    }

    // --- Targets (optional) ---
    if (normalize_targets) {
        size_t t_cols = Y_.cols();
        min_Y_ = Matrix(1, t_cols);
        max_Y_ = Matrix(1, t_cols);

        for (size_t j = 0; j < t_cols; ++j) {
            double min_val = Y_(0, j), max_val = Y_(0, j);
            for (size_t i = 1; i < rows; ++i) {
                min_val = std::min(min_val, Y_(i, j));
                max_val = std::max(max_val, Y_(i, j));
            }
            min_Y_(0, j) = min_val;
            max_Y_(0, j) = max_val;
            double range = (max_val - min_val == 0.0) ? 1.0 : (max_val - min_val);

            for (size_t i = 0; i < rows; ++i)
                Y_(i, j) = ((Y_(i, j) - min_val) / range) * (new_max - new_min) + new_min;
        }
    }

    norm_type_ = NormType::MINMAX;
}

inline Dataset Dataset::normalize_minmax(double new_min = 0.0, double new_max = 1.0, bool normalize_targets = false) {
    Dataset temp(*this);
    temp.normalize_minmax_self(new_min, new_max, normalize_targets);
    return temp;
}

// Denormalize X and Y separately for min-max
inline Matrix Dataset::denormalize_minmax_X(const Matrix& normalized_X, double orig_min = 0.0, double orig_max = 1.0) const {
    Matrix result = normalized_X;
    for (size_t j = 0; j < result.cols(); ++j) {
        double min_val = min_X_(0, j);
        double max_val = max_X_(0, j);
        double range = (max_val - min_val == 0.0) ? 1.0 : (max_val - min_val);

        for (size_t i = 0; i < result.rows(); ++i)
            result(i, j) = ((result(i, j) - orig_min) / (orig_max - orig_min)) * range + min_val;
    }
    return result;
}

inline Matrix Dataset::denormalize_minmax_Y(const Matrix& normalized_Y, double orig_min = 0.0, double orig_max = 1.0) const {
    Matrix result = normalized_Y;
    for (size_t j = 0; j < result.cols(); ++j) {
        double min_val = min_Y_(0, j);
        double max_val = max_Y_(0, j);
        double range = (max_val - min_val == 0.0) ? 1.0 : (max_val - min_val);

        for (size_t i = 0; i < result.rows(); ++i)
            result(i, j) = ((result(i, j) - orig_min) / (orig_max - orig_min)) * range + min_val;
    }
    return result;
}

// Utilities
inline bool Dataset::load_csv(const std::string& filename, bool has_header = true, size_t num_target = 1) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels;
    std::string line;

    if (has_header) {
        std::getline(file, line);
        std::stringstream header_ss(line);
        std::string cell;
        std::vector<std::string> headers;

        while (std::getline(header_ss, cell, ',')) headers.push_back(cell);
        feature_names_ = std::vector<std::string>(headers.begin(), headers.end() - num_target);
        target_names_ = std::vector<std::string>(headers.end() - num_target, headers.end());
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row_values;

        while (std::getline(ss, cell, ','))
            row_values.push_back(std::stod(cell));

        labels.push_back(std::vector<double>(row_values.end() - num_target, row_values.end()));
        row_values.resize(row_values.size() - num_target);  // removes targets 

        features.push_back(row_values);
    }

    size_t rows = features.size();
    size_t cols = features[0].size();
    X_ = Matrix(rows, cols);
    Y_ = Matrix(rows, labels[0].size());

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j)
            X_(i, j) = features[i][j];
        for (size_t j = 0; j < labels[0].size(); ++j)
            Y_(i, j) = labels[i][j];
    }

    file.close();
    norm_type_ = NormType::NONE;
    return true;
}

inline Dataset Dataset::shuffle(size_t seed = 42) const {
    size_t n_samples = X_.rows();
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    Dataset shuffled_set;
    shuffled_set.X_ = Matrix(n_samples, X_.cols());
    shuffled_set.Y_ = Matrix(n_samples, Y_.cols());
    for (size_t i = 0; i < n_samples; ++i) {
        size_t idx = indices[i];
        for (size_t j = 0; j < X_.cols(); ++j)
            shuffled_set.X_(i, j) = X_(idx, j);
        for (size_t j = 0; j < Y_.cols(); ++j)
            shuffled_set.Y_(i, j) = Y_(idx, j);
    }
    return shuffled_set;
}

inline void Dataset::shuffle_self(size_t seed = 42) {
    *this = shuffle(seed);
}

// Train and test split (set seed = 0 for no shuffle)
inline std::pair<Dataset, Dataset> Dataset::train_test_split(double train_ratio = 0.5, unsigned seed = 0) const {
    if (X_.rows() != Y_.rows())
        throw std::runtime_error("Feature and label size mismatch.");

    Dataset temp = *this;
    size_t n_samples = X_.rows();
    if (seed != 0) {
        temp.shuffle_self(seed);
    }

    size_t train_size = static_cast<size_t>(n_samples * train_ratio);
    size_t test_size = n_samples - train_size;

    Dataset train_set, test_set;
    train_set.X_ = Matrix(train_size, X_.cols());
    train_set.Y_ = Matrix(train_size, Y_.cols());
    test_set.X_ = Matrix(test_size, X_.cols());
    test_set.Y_ = Matrix(test_size, Y_.cols());

    for (size_t i = 0; i < train_size; ++i) {
        for (size_t j = 0; j < temp.X_.cols(); ++j)
            train_set.X_(i, j) = temp.X_(i, j);
        for (size_t j = 0; j < temp.Y_.cols(); ++j)
            train_set.Y_(i, j) = temp.Y_(i, j);
    }

    for (size_t i = 0; i < test_size; ++i) {
        for (size_t j = 0; j < temp.X_.cols(); ++j)
            test_set.X_(i, j) = temp.X_(train_size + i, j);
        for (size_t j = 0; j < temp.Y_.cols(); ++j)
            test_set.Y_(i, j) = temp.Y_(train_size + i, j);
    }
    return { train_set, test_set };
}

