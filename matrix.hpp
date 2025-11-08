#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <thread>
#include <future>
#include <numeric>
#include <limits>
#include <random>

namespace MATRIX_CONFIGS {
    static bool ENABLE_CHECKS = false;
    static bool ENABLE_MULTITHREADING = false;
    static size_t MULTITHREADING_THRESHOLD = 500;  // Use multithreading (even if enabled) only if range is greater or equal to this
    static size_t NUM_THREADS = 1;
}

using namespace MATRIX_CONFIGS;

class Matrix {
private:
    std::vector<double> data_;
    size_t rows_;
    size_t cols_;

    // Private helper methods
    void validate_dimensions(const Matrix& other) const;
    void validate_multiplication(const Matrix& other) const;
    void check_nan_inf() const;

public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols, double init_val = 0.0);
    Matrix(size_t rows, size_t cols, const std::vector<double>& values);

    // Accessors
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    size_t rows() const;
    size_t cols() const;
    size_t size() const;
    void set_num_threads(size_t n);
    void set_enable_checks(bool enable);
    void set_enable_multithreading(bool enable);
    void set_multithreading_threshold(size_t n);

    // Element-wise operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix element_wise_multiply(const Matrix& other) const;
    Matrix element_wise_divide(const Matrix& other) const;

    // Scalar operations
    Matrix operator+(double scalar) const;
    Matrix operator-(double scalar) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    // In-place operations
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);

    // Matrix operations
    Matrix transpose() const;
    Matrix transpose_parallel_blocked() const;
    Matrix row(size_t idx) const;
    Matrix col(size_t idx) const;
    Matrix sub_matrix(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const;
    Matrix sub_matrix(size_t row_end, size_t col_end) const;

    // Statistical operations
    double min() const;
    double max() const;
    double sum() const;
    double mean() const;
    double variance() const;
    double std_dev() const;

    // Element-wise math functions
    Matrix sqrt() const;
    Matrix exp() const;
    Matrix log() const;
    Matrix pow(double exponent) const;
    Matrix abs() const;
    Matrix sin() const;
    Matrix cos() const;
    Matrix tan() const;
    Matrix sum_rowwise() const;
    Matrix sum_colwise() const;

    // Utility functions
    void fill(double value);
    Matrix randomize(double min = 0.0, double max = 1.0);
    void randomize_inplace(double min = 0.0, double max = 1.0);
    Matrix reshape(size_t new_rows, size_t new_cols) const;
    Matrix broadcast_to(size_t target_rows, size_t target_cols) const;
    const std::vector<double>& get_data() const;
    std::vector<double>& get_data();

    // Utility for multithreading
    template<typename Func>
    static void parallel_for(size_t start, size_t end, Func&& func);
};

// ==================== IMPLEMENTATIONS ====================

// Private helper methods
inline void Matrix::validate_dimensions(const Matrix& other) const {
    if (!ENABLE_CHECKS) return;    
    if (!(rows_ == other.rows_ && cols_ == other.cols_))
        throw std::invalid_argument("Matrix dimensions not same");

}

inline void Matrix::validate_multiplication(const Matrix& other) const {
    if (ENABLE_CHECKS && cols_ != other.rows_) {
        throw std::invalid_argument("Invalid dimensions for multiplication");
    }
}

inline void Matrix::check_nan_inf() const {
    if (!ENABLE_CHECKS) return;
    for (const auto& val : data_) {
        if (std::isnan(val) || std::isinf(val)) {
            throw std::runtime_error("Matrix contains NaN or Inf values");
        }
    }
}

template<typename Func>
static inline void Matrix::parallel_for(size_t start, size_t end, Func&& func) {
    if (!ENABLE_MULTITHREADING || NUM_THREADS <= 1 || (end - start) <= MULTITHREADING_THRESHOLD) {
        func(start, end);
        return;
    }

    size_t chunk_size = (end - start + NUM_THREADS - 1) / NUM_THREADS;
    std::vector<std::future<void>> futures;
    futures.reserve(NUM_THREADS);

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        size_t chunk_start = start + i * chunk_size;
        size_t chunk_end = std::min(chunk_start + chunk_size, end);
        if (chunk_start >= end) break;

        futures.push_back(std::async(std::launch::async, func, chunk_start, chunk_end));
    }

    for (auto& f : futures)
        f.get();
}


inline Matrix::Matrix()
    : data_(1, 0.0), rows_(1), cols_(1) { }

// Constructors
inline Matrix::Matrix(size_t rows, size_t cols, double init_val)
    : data_(rows* cols, init_val), rows_(rows), cols_(cols) { }

inline Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& values)
    : data_(values), rows_(rows), cols_(cols) {
    if (ENABLE_CHECKS && data_.size() != rows_ * cols_) {
        throw std::invalid_argument("Vector size must match matrix dimensions");
    }
}

// Accessors
inline double& Matrix::operator()(size_t row, size_t col) {
    return data_[row * cols_ + col];
}

inline const double& Matrix::operator()(size_t row, size_t col) const {
    return data_[row * cols_ + col];
}

inline size_t Matrix::rows() const { return rows_; }
inline size_t Matrix::cols() const { return cols_; }
inline size_t Matrix::size() const { return data_.size(); }
inline void Matrix::set_num_threads(size_t n) {
    NUM_THREADS = n;
    ENABLE_MULTITHREADING = (n > 1);
}
inline void Matrix::set_enable_checks(bool enable) { ENABLE_CHECKS = enable; }
inline void Matrix::set_enable_multithreading(bool enable) { ENABLE_MULTITHREADING = enable; }
inline void Matrix::set_multithreading_threshold(size_t n) { MULTITHREADING_THRESHOLD = n; }

// Element-wise operations
inline Matrix Matrix::operator+(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        });

    return result;
}

inline Matrix Matrix::operator-(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        });

    return result;
}

inline Matrix Matrix::operator*(const Matrix& other) const {
    validate_multiplication(other);
    Matrix result(rows_, other.cols_, 0.0);

    parallel_for(0, rows_, [&](size_t start_row, size_t end_row) {
        for (size_t i = start_row; i < end_row; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        });

    return result;
}

inline Matrix Matrix::element_wise_multiply(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        });

    return result;
}

inline Matrix Matrix::element_wise_divide(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            if (ENABLE_CHECKS && other.data_[i] == 0.0) {
                throw std::runtime_error("Division by zero");
            }
            result.data_[i] = data_[i] / other.data_[i];
        }
        });

    return result;
}

// Scalar operations
inline Matrix Matrix::operator+(double scalar) const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] + scalar;
        }
        });

    return result;
}

inline Matrix Matrix::operator-(double scalar) const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] - scalar;
        }
        });

    return result;
}

inline Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        });

    return result;
}

inline Matrix Matrix::operator/(double scalar) const {
    if (ENABLE_CHECKS && scalar == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = data_[i] / scalar;
        }
        });

    return result;
}

// In-place operations
inline Matrix& Matrix::operator+=(const Matrix& other) {
    validate_dimensions(other);
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data_[i] += other.data_[i];
        }
        });
    return *this;
}

inline Matrix& Matrix::operator-=(const Matrix& other) {
    validate_dimensions(other);
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data_[i] -= other.data_[i];
        }
        });
    return *this;
}

inline Matrix& Matrix::operator*=(double scalar) {
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data_[i] *= scalar;
        }
        });
    return *this;
}

inline Matrix& Matrix::operator/=(double scalar) {
    if (ENABLE_CHECKS && scalar == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data_[i] /= scalar;
        }
        });
    return *this;
}

// Matrix operations
inline Matrix Matrix::transpose() const {
    if (ENABLE_MULTITHREADING) {
        return transpose_parallel_blocked();
    }
    Matrix result(cols_, rows_, 0.0);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

inline Matrix Matrix::transpose_parallel_blocked() const {
    Matrix result(cols_, rows_, 0.0);

    size_t mid_row = rows_ / 2;
    size_t mid_col = cols_ / 2;

    auto transpose_block = [&](size_t r_start, size_t r_end, size_t c_start, size_t c_end) {
        if (r_start >= r_end || c_start >= c_end) return;
        for (size_t i = r_start; i < r_end; ++i) {
            for (size_t j = c_start; j < c_end; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        };

    if (!ENABLE_MULTITHREADING || NUM_THREADS <= 1) {
        // Serial version
        transpose_block(0, mid_row, 0, mid_col);          // A11
        transpose_block(0, mid_row, mid_col, cols_);      // A12
        transpose_block(mid_row, rows_, 0, mid_col);      // A21
        transpose_block(mid_row, rows_, mid_col, cols_);  // A22
    }
    else {
        // Parallelized version using parallel_for()
        parallel_for(0, 4, [&](size_t start, size_t end) {
            for (size_t block = start; block < end; ++block) {
                switch (block) {
                case 0: transpose_block(0, mid_row, 0, mid_col); break;          // A11
                case 1: transpose_block(0, mid_row, mid_col, cols_); break;      // A12
                case 2: transpose_block(mid_row, rows_, 0, mid_col); break;      // A21
                case 3: transpose_block(mid_row, rows_, mid_col, cols_); break;  // A22
                default: break;
                }
            }
            });
    }

    return result;
}

inline Matrix Matrix::row(size_t idx) const {
    if (ENABLE_CHECKS && idx >= rows_) {
        throw std::out_of_range("Row index out of range");
    }
    std::vector<double> row_data(data_.begin() + idx * cols_, data_.begin() + (idx + 1) * cols_);
    return Matrix(1, cols_, row_data);
}

inline Matrix Matrix::col(size_t idx) const {
    if (ENABLE_CHECKS && idx >= cols_) {
        throw std::out_of_range("Column index out of range");
    }
    std::vector<double> col_data(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        col_data[i] = (*this)(i, idx);
    }
    return Matrix(rows_, 1, col_data);
}

inline Matrix Matrix::sub_matrix(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const {
    if (ENABLE_CHECKS) {
        if (row_start >= rows_ || row_end >= rows_ || row_start > row_end)
            throw std::out_of_range("Row indices out of range");
        if (col_start >= cols_ || col_end >= cols_ || col_start > col_end)
            throw std::out_of_range("Column indices out of range");
    }

    size_t new_rows = row_end - row_start + 1;
    size_t new_cols = col_end - col_start + 1;
    Matrix result(new_rows, new_cols);

    auto& src = data_;
    auto& dst = result.get_data();

    parallel_for(0, new_rows, [&](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            size_t src_offset = (r + row_start) * cols_ + col_start;
            size_t dst_offset = r * new_cols;
            std::copy(src.begin() + src_offset, src.begin() + src_offset + new_cols, dst.begin() + dst_offset);
        }
        });

    return result;
}

inline Matrix Matrix::sub_matrix(size_t row_end, size_t col_end) const {
    return sub_matrix(0, row_end, 0, col_end);
}


// Statistical operations
inline double Matrix::min() const {
    if (!ENABLE_MULTITHREADING || NUM_THREADS <= 1) {
        return *std::min_element(data_.begin(), data_.end());
    }

    size_t chunk_size = (data_.size() + NUM_THREADS - 1) / NUM_THREADS;
    std::vector<std::future<double>> futures;

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, data_.size());
        if (start >= data_.size()) break;

        futures.push_back(std::async(std::launch::async, [this, start, end]() {
            return *std::min_element(data_.begin() + start, data_.begin() + end);
            }));
    }

    double min_val = std::numeric_limits<double>::max();
    for (auto& f : futures) {
        min_val = std::min(min_val, f.get());
    }
    return min_val;
}

inline double Matrix::max() const {
    if (!ENABLE_MULTITHREADING || NUM_THREADS <= 1) {
        return *std::max_element(data_.begin(), data_.end());
    }

    size_t chunk_size = (data_.size() + NUM_THREADS - 1) / NUM_THREADS;
    std::vector<std::future<double>> futures;

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, data_.size());
        if (start >= data_.size()) break;

        futures.push_back(std::async(std::launch::async, [this, start, end]() {
            return *std::max_element(data_.begin() + start, data_.begin() + end);
            }));
    }

    double max_val = std::numeric_limits<double>::lowest();
    for (auto& f : futures) {
        max_val = std::max(max_val, f.get());
    }
    return max_val;
}

inline double Matrix::sum() const {
    if (!ENABLE_MULTITHREADING || NUM_THREADS <= 1) {
        return std::accumulate(data_.begin(), data_.end(), 0.0);
    }

    size_t chunk_size = (data_.size() + NUM_THREADS - 1) / NUM_THREADS;
    std::vector<std::future<double>> futures;

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, data_.size());
        if (start >= data_.size()) break;

        futures.push_back(std::async(std::launch::async, [this, start, end]() {
            return std::accumulate(data_.begin() + start, data_.begin() + end, 0.0);
            }));
    }

    double total = 0.0;
    for (auto& f : futures) {
        total += f.get();
    }
    return total;
}

inline double Matrix::mean() const {
    return sum() / static_cast<double>(data_.size());
}

inline double Matrix::variance() const {
    double m = mean();
    double var_sum = 0.0;

    if (!ENABLE_MULTITHREADING || NUM_THREADS <= 1) {
        for (const auto& val : data_) {
            double diff = val - m;
            var_sum += diff * diff;
        }
    }
    else {
        size_t chunk_size = (data_.size() + NUM_THREADS - 1) / NUM_THREADS;
        std::vector<std::future<double>> futures;

        for (size_t i = 0; i < NUM_THREADS; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, data_.size());
            if (start >= data_.size()) break;

            futures.push_back(std::async(std::launch::async, [this, start, end, m]() {
                double local_sum = 0.0;
                for (size_t j = start; j < end; ++j) {
                    double diff = data_[j] - m;
                    local_sum += diff * diff;
                }
                return local_sum;
                }));
        }

        for (auto& f : futures) {
            var_sum += f.get();
        }
    }

    return var_sum / static_cast<double>(data_.size());
}

inline double Matrix::std_dev() const {
    return std::sqrt(variance());
}

// Element-wise math functions
inline Matrix Matrix::sqrt() const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::sqrt(data_[i]);
        }
        });

    if (ENABLE_CHECKS) result.check_nan_inf();
    return result;
}

inline Matrix Matrix::exp() const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::exp(data_[i]);
        }
        });

    if (ENABLE_CHECKS) result.check_nan_inf();
    return result;
}

inline Matrix Matrix::log() const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::log(data_[i]);
        }
        });

    if (ENABLE_CHECKS) result.check_nan_inf();
    return result;
}

inline Matrix Matrix::pow(double exponent) const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::pow(data_[i], exponent);
        }
        });

    if (ENABLE_CHECKS) result.check_nan_inf();
    return result;
}

inline Matrix Matrix::abs() const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::abs(data_[i]);
        }
        });

    return result;
}

inline Matrix Matrix::sin() const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::sin(data_[i]);
        }
        });

    return result;
}

inline Matrix Matrix::cos() const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::cos(data_[i]);
        }
        });

    return result;
}

inline Matrix Matrix::tan() const {
    Matrix result(rows_, cols_, 0.0);

    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            result.data_[i] = std::tan(data_[i]);
        }
        });

    if (ENABLE_CHECKS) result.check_nan_inf();
    return result;
}

inline Matrix Matrix::sum_rowwise() const {
    // Sum along rows → result is 1 x cols
    Matrix result(1, cols_);
    parallel_for(0, cols_, [&](size_t start, size_t end) {
        for (size_t j = start; j < end; ++j) {
            for (size_t i = 0; i < rows_; ++i)
                result(0, j) += (*this)(i, j);
        }});
    return result;
}

inline Matrix Matrix::sum_colwise() const {
    // Sum along cols → result is rows x 1
    Matrix result(rows_, 1);
    parallel_for(0, rows_, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            for (size_t j = 0; j < cols_; ++j)
                result(i, 0) += (*this)(i, j);
        }});
        return result;
}

// Utility functions
inline void Matrix::fill(double value) {
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        std::fill(data_.begin() + start, data_.begin() + end, value);
        });
}
inline Matrix Matrix::randomize(double min, double max)
{
    if (ENABLE_CHECKS && min > max) {
        throw std::invalid_argument("min value cannot be greater than max value");
    }
    Matrix result(rows_, cols_);
    std::random_device rd;

    parallel_for(0, result.data_.size(), [&](size_t start, size_t end) {
        // Each thread gets its own generator to avoid data races
        thread_local std::mt19937 thread_gen(rd() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::uniform_real_distribution<double> thread_dist(min, max);

        for (size_t i = start; i < end; ++i) {
            result.data_[i] = thread_dist(thread_gen);
        }
        });
    
    return result;
}
inline void Matrix::randomize_inplace(double min, double max)
{
    if (ENABLE_CHECKS && min > max) {
        throw std::invalid_argument("min value cannot be greater than max value");
    }

    std::random_device rd;
    parallel_for(0, data_.size(), [&](size_t start, size_t end) {
        // Each thread gets its own generator to avoid data races
        thread_local std::mt19937 thread_gen(rd() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::uniform_real_distribution<double> thread_dist(min, max);

        for (size_t i = start; i < end; ++i) {
            data_[i] = thread_dist(thread_gen);
        }
        });
}

inline Matrix Matrix::reshape(size_t new_rows, size_t new_cols) const {
    if (ENABLE_CHECKS && new_rows * new_cols != data_.size()) {
        throw std::invalid_argument("New dimensions must match total size");
    }
    return Matrix(new_rows, new_cols, data_);
}

inline Matrix Matrix::broadcast_to(size_t target_rows, size_t target_cols) const
{
    if (ENABLE_CHECKS) {
        if (rows_ != target_rows && rows_ != 1)
            throw std::invalid_argument("Cannot broadcast: incompatible row dimension");
        if (cols_ != target_cols && cols_ != 1)
            throw std::invalid_argument("Cannot broadcast: incompatible column dimension");
    }
    if (rows_ == target_rows && cols_ == target_cols)
        return Matrix(*this);

    Matrix result(target_rows, target_cols);
    if (rows_ == 1) {
        parallel_for(0, target_rows, [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i)
                for (size_t j = 0; j < target_cols; ++j)
                    result(i, j) = (*this)(0, j);
            });
    }
    else if (cols_ == 1) {
        parallel_for(0, target_rows, [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i)
                for (size_t j = 0; j < target_cols; ++j)
                    result(i, j) = (*this)(i, 0);
            });
    }

    return result;
}


inline const std::vector<double>& Matrix::get_data() const { return data_; }
inline std::vector<double>& Matrix::get_data() { return data_; }

#endif // MATRIX_HPP