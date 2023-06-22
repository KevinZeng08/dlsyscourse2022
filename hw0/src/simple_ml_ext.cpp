#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    const int num_examples = m;
    const int num_classes = k;
    const int input_dim = n;

    // SGD by batches
    for(int i=0; i<num_examples;i+=batch) {
      int batch_size = std::min(static_cast<int>(batch), num_examples - i);
      const float *X_batch = X + i * input_dim;
      const unsigned char *y_batch = y + i;

      // Allocate memory
      float *Z = new float[batch_size * num_classes]();
      float *Iy = new float[batch_size * num_classes]();

      // Compute normalized logits and one-hot bases
      // Z = np.exp(np.dot(X_batch,theta))
      // Z[j,k]=X_batch的第j行 * theta的第k列
      for(int j=0;j<batch_size;j++){
        float row_sum = 0.0;
        for(int k=0;k<num_classes;k++){
          float dot_prod = 0.0;
          for(int l=0;l<input_dim;l++){
            dot_prod += X_batch[j*input_dim + l] * theta[l*num_classes + k];
          }
          Z[j*num_classes + k] = std::exp(dot_prod);
          row_sum += Z[j*num_classes + k];
        }
        // normalization applied row-wise
        // Z /= np.sum(Z, axis=1, keepdims=True)
        for(int k=0;k<num_classes;k++){
          Z[j*num_classes+k] /= row_sum;
          // one-hot bases for y_batch Iy[i,y[i]] = 1
          // Iy = np.zeros((batch, num_classes), dtype=np.float32)
          // Iy[np.arange(batch), y_batch] = 1;
          if(y_batch[j] == k) {
            Iy[j*num_classes + k] = 1;
          }
        }
      }

      // Compute gradient
      // grad = np.dot(X_batch.T, Z-Iy) / batch
      // grad[j][k] = X_batch.T的第j行(等价于X_batch的第j列) * Z-Iy的第k列 
      float *grad = new float[input_dim * num_classes]();
      for(int j=0;j<input_dim;j++){
        for(int k=0;k<num_classes;k++){
          for(int l=0;l<batch_size;l++){
            grad[j*num_classes + k] += X_batch[l*input_dim + j] * (Z[l*num_classes + k]-Iy[l*num_classes+k]);
          }
          grad[j*num_classes + k] /= batch_size;
        }
      }
      
      // Update theta by learning rate and gradient
      // theta -= lr * grad
      for(int j=0;j<input_dim;j++){
        for(int k=0;k<num_classes;k++){
          theta[j*num_classes+k] -= lr * grad[j*num_classes + k];
        }
      }

      delete Z;
      delete Iy;
      delete grad;
    }
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
