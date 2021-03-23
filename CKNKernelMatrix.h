#include <cassert>
#include <cmath>
// #include <glog/logging.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


#define FOR2DTO(ii, jj, hh, ww)          \
  for (int ii = 0; ii < hh; ++ii)        \
    for (int jj = 0; jj < ww; ++jj)      \

#define INIT_TIME                                 \
    static thread_local sclock::time_point start; \
    static thread_local std::chrono::duration<double> elapsed;

#define TIC                 \
  if (verbose) {            \
    start = sclock::now();  \
  }

#define TOC(label, ...) \
  if (verbose) { \
    elapsed = sclock::now() - start; \
    std::cerr << label ": " << elapsed.count() << " " __VA_ARGS__; \
  }

using sclock = std::chrono::system_clock;

namespace {
static const double PI = 3.141592653589793238463;

}

enum PoolType {
  GAUSSIAN = 0,
  STRIDED,
  AVERAGE
};

enum KernelType {
  EXP = 0,
  RELU,
  LINEAR,
  POLY2,
  POLY3,
  POLY4,
  SQUARE  // non-homogeneous square
};

struct LayerParams {
  size_t patchSize;
  size_t subsampling;
  size_t poolFactor;
  KernelType kernelType;
  double kernelParam; // sigma for EXP kernel
  bool zeroPad;
  PoolType poolType;
};

template <typename Double>
class CKNKernelMatrixEigen {
 public:
  CKNKernelMatrixEigen(const std::vector<LayerParams>& layers,
                       const size_t h,
                       const size_t w,
                       const size_t c,
                       const bool verbose = false)
    : layers_(layers), h_(h), w_(w), c_(c), verbose_(verbose) {
    // CHECK(h == w) << "only square images are supported for now";
    assert(h == w);
    layerDims_.resize(layers.size());
    layerDims_[0].hi = h;
    layerDims_[0].wi = w;

    poolFilter_.resize(layers.size());

    for (size_t l = 0; l < layers.size(); ++l) {
      auto& dims = layerDims_[l];
      if (l > 0) {
        dims.hi = layerDims_[l-1].hpool;
        dims.wi = layerDims_[l-1].wpool;
      }
      const size_t patch = layers_[l].patchSize;
      if (layers_[l].zeroPad) {
        dims.hconv = dims.hi;
        dims.wconv = dims.wi;
      } else {
        assert(false); // only zeroPad is implemented (TODO)
        // CHECK_GE(dims.hi, patch);
        assert(dims.hi >= patch);
        dims.hconv = dims.hi - patch + 1;
        dims.wconv = dims.wi - patch + 1;
      }
      const size_t sub = layers_[l].subsampling;
      const size_t poolFactor = layers_[l].poolFactor;
      if (layers_[l].poolType == GAUSSIAN) {
        dims.poolSize = 2 * poolFactor + 1;
        poolFilter_[l] = makeFilter(dims.poolSize, layers_[l].poolType);

        // sample at sub * i, with i = 1..hpool
        // CHECK_GE(dims.hconv, sub) << "too large subsampling";
        assert(dims.hconv >= sub);
        dims.hpool = dims.hconv / sub - 1;
        dims.wpool = dims.wconv / sub - 1;
      } else if (layers_[l].poolType == AVERAGE) {
        dims.poolSize = poolFactor;
        poolFilter_[l] = makeFilter(dims.poolSize, layers_[l].poolType);
        dims.hpool = dims.hconv / sub;
        dims.wpool = dims.wconv / sub;
      } else if (layers_[l].poolType == STRIDED) {
        dims.poolSize = poolFactor;
        dims.hpool = dims.hconv / sub - 1;
        dims.wpool = dims.wconv / sub - 1;
      } else {
        assert(false);
      }

      if (verbose_) {
        // LOG(INFO) << "layer " << l << "(" << patch << "," << sub
        std::clog << "layer " << l << "(" << patch << "," << sub
                  << "): " << dims.hi << "x" << dims.wi << " -> " << dims.hconv
                  << "x" << dims.wconv << " -> " << dims.hpool << "x"
                  << dims.wpool << " \n";
      }
    }
  }

  size_t nlayers() const {
    return layers_.size();
  }

  void computeNorms(const Double* im,
                    Double* norms,
                    Double* normsInv,
                    const bool verbose = false) {
    const Eigen::Map<const Matrix> x(im, h_ * w_, c_);

    INIT_TIME;
    TIC;
    Matrix xx = x * x.transpose();
    TOC("init pool");

    Tensor pooled;

    for (size_t l = 0; l < nlayers(); ++l) {
      TIC;
      Tensor patch = toPatches((l == 0 ? xx.data() : pooled.data()), l);
      TOC("patch");

      TIC;
      const size_t h = layerDims_[l].hconv;
      const size_t w = layerDims_[l].wconv;
      Eigen::Map<Matrix> patchMap(patch.data(), h * w, h * w);
      Eigen::Map<Array> normMap(norms, h * w);
      Eigen::Map<Array> normInvMap(normsInv, h * w);
      normMap = patchMap.diagonal().array().sqrt();
      normInvMap = normMap.array().max(1e-6).inverse();
      TOC("norm");

      if (l == nlayers() - 1) {
        break; // rest not needed for final layer
      }
      TIC;
      Array2d cosine = cosines(patch.data(), normsInv, normsInv, l);
      TOC("cosine");

      TIC;
      Array2d mapped = kappa1(cosine.data(), norms, norms, l);
      TOC("kappa1");

      TIC;
      pooled = pool(mapped.data(), l);
      TOC("pool");

      // move to next layer norms
      norms += h_ * w_;
      normsInv += h_ * w_;
    }

    if (verbose) {
      std::cerr << std::endl;
    }
  }

  Double computeKernel(const Double* im1,
                       const Double* im2,
                       const Double* norms1,
                       const Double* norms2,
                       const Double* normsInv1,
                       const Double* normsInv2,
                       const bool useNtk = false,
                       const bool verbose = false) {
    const Eigen::Map<const Matrix> x(im1, h_ * w_, c_);
    const Eigen::Map<const Matrix> y(im2, h_ * w_, c_);

    INIT_TIME;
    TIC;
    Matrix xy = x * y.transpose();
    TOC("init pool");

    Tensor pooled, pooledNtk;

    for (size_t l = 0; l < nlayers(); ++l) {
      TIC;
      Tensor patch = toPatches((l == 0 ? xy.data() : pooled.data()), l);
      TOC("patch");

      Tensor patchNtk;
      if (useNtk && l >= 1) {
        TIC;
        patchNtk = toPatches(pooledNtk.data(), l);
        TOC("patchNTK");
      }

      TIC;
      Array2d cosine = cosines(patch.data(), normsInv1, normsInv2, l);
      TOC("cosine");

      TIC;
      Array2d mapped = kappa1(cosine.data(), norms1, norms2, l);
      TOC("kappa1");

      Array2d mappedNtk;
      if (useNtk) {
        TIC;
        mappedNtk = ukappa0(cosine.data(), (l == 0) ? patch.data() : patchNtk.data(), l);
        mappedNtk += mapped;
        TOC("ukappa0");
      }

      TIC;
      pooled = pool(mapped.data(), l);
      TOC("pool");

      if (useNtk) {
        TIC;
        pooledNtk = pool(mappedNtk.data(), l);
        TOC("poolNTK");
      }

      // move to next layer norms
      norms1 += h_ * w_;
      norms2 += h_ * w_;
      normsInv1 += h_ * w_;
      normsInv2 += h_ * w_;
    }
    if (verbose) {
      std::cerr << std::endl;
    }

    const size_t h = layerDims_[nlayers() - 1].hpool;
    const size_t w = layerDims_[nlayers() - 1].wpool;

    if (useNtk) { // cache result for non-NTK kernel when computing NTK
      Eigen::Map<const Matrix> pooledMap(pooled.data(), h * w, h * w);
      cachedRF_ = pooledMap.trace();
    }

    Eigen::Map<const Matrix> pooledMap(
        useNtk ? pooledNtk.data() : pooled.data(), h * w, h * w);
    return pooledMap.trace();
  }

  Double cachedRFKernel() const {
    return cachedRF_;
  }

 private:
  using Matrix =
      Eigen::Matrix<Double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Array2d =
      Eigen::Array<Double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector =
      Eigen::Matrix<Double, Eigen::Dynamic, 1>;
  using Array =
      Eigen::Array<Double, Eigen::Dynamic, 1>;
  using Tensor =
      Eigen::Tensor<Double, 4, Eigen::RowMajor>;

  // kernel after patch extraction
  Tensor toPatches(Double* data, const int32_t l) const {
    const int sz = layers_[l].patchSize;
    const size_t hi = layerDims_[l].hi;
    const size_t wi = layerDims_[l].wi;
    const size_t h = layerDims_[l].hconv;
    const size_t w = layerDims_[l].wconv;
    const Eigen::TensorMap<const Tensor> pooled(data, hi, wi, hi, wi);

    if (sz == 1) {
      Tensor out = pooled;
      return out;
    }

    const int start = -(sz - 1) / 2;

    Tensor patch1 = pooled;
    for (int r = start; r < start + sz; ++r) {
      if (r == 0) { // done at init
        continue;
      }
      const int rplus = std::max(0, r);
      const int rminus = -std::min(0, r);
      const int hlen = h - rminus - rplus;
      for (int i1 = rplus; i1 < h - rminus; ++i1) {
        for (int j1 = 0; j1 < w; ++j1) {
          Eigen::Map<Array> out(&patch1(i1, j1, rplus, 0), hlen * w);
          Eigen::Map<const Array> block(
              &pooled(i1 - r, j1, rminus, 0), hlen * w);
          out += block;
        }
      }
    }

    Tensor patch = patch1;
    for (int r = start; r < start + sz; ++r) {
      if (r == 0) { // done at init
        continue;
      }
      const int rplus = std::max(0, r);
      const int rminus = std::max(0, -r);
      const int wlen = w - rminus - rplus;
      for (int i1 = 0; i1 < h; ++i1) {
        for (int j1 = rplus; j1 < h - rminus; ++j1) {
          for (int i2 = 0; i2 < h; ++i2) {
            Eigen::Map<Array> out(&patch(i1, j1, i2, rplus), wlen);
            Eigen::Map<const Array> block(
                &patch1(i1, j1 - r, i2, rminus), wlen);
            out += block;
          }
        }
      }
    }
    return patch;
  }

  Array2d cosines(const Double* const data,
                  const Double* const normsInvL,
                  const Double* const normsInvR,
                  const int32_t l) const {
    const size_t h = layerDims_[l].hconv;
    const size_t w = layerDims_[l].wconv;
    Eigen::Map<const Array> normInvLMap(normsInvL, h * w);
    Eigen::Map<const Array> normInvRMap(normsInvR, h * w);
    Eigen::Map<const Array2d> patchMap(data, h * w, h * w);
    Array2d cosine = patchMap.colwise() * normInvLMap;
    cosine.rowwise() *= normInvRMap.transpose();
    return cosine;
  }

  Array2d kappa1(const Double* const cosines,
                 const Double* const normsL,
                 const Double* const normsR,
                 const int32_t l) const {
    const size_t h = layerDims_[l].hconv;
    const size_t w = layerDims_[l].wconv;
    Eigen::Map<const Array> normLMap(normsL, h * w);
    Eigen::Map<const Array> normRMap(normsR, h * w);
    Eigen::Map<const Array2d> cosMap(cosines, h * w, h * w);
    if (layers_[l].kernelType == EXP) {
      const Double sigma = static_cast<Double>(layers_[l].kernelParam);
      const Double alpha = 1. / (sigma * sigma);
      Array2d out = alpha * (cosMap - 1.);
      out = out.exp();
      out.colwise() *= normLMap;
      out.rowwise() *= normRMap.transpose();
      return out;
    } else if (layers_[l].kernelType == RELU) {
      Array2d cos = cosMap.min(1.0);
      // Array2d out = cos * (PI - cos.acos()) + (1. - cos.square()).sqrt();
      Array2d theta = cos.acos();
      Array2d out = cos.square();
      out = 1. - out;
      out = out.sqrt();
      out += cos * (static_cast<Double>(PI) - theta);

      out /= PI;
      out.colwise() *= normLMap;
      out.rowwise() *= normRMap.transpose();
      return out;
    } else if (layers_[l].kernelType == LINEAR) {
      const Double factor = static_cast<Double>(layers_[l].kernelParam);
      Array2d out = factor * cosMap;
      out.colwise() *= normLMap;
      out.rowwise() *= normRMap.transpose();
      return out;
    } else if (layers_[l].kernelType == POLY2) {
      const Double factor = static_cast<Double>(layers_[l].kernelParam);
      Array2d out = cosMap.square();
      out.colwise() *= normLMap;
      out.rowwise() *= normRMap.transpose();
      return out;
    } else if (layers_[l].kernelType == POLY3) {
      const Double factor = static_cast<Double>(layers_[l].kernelParam);
      Array2d out = cosMap * cosMap.square();
      out.colwise() *= normLMap;
      out.rowwise() *= normRMap.transpose();
      return out;
    } else if (layers_[l].kernelType == POLY4) {
      const Double factor = static_cast<Double>(layers_[l].kernelParam);
      Array2d out = cosMap.square().square();
      out.colwise() *= normLMap;
      out.rowwise() *= normRMap.transpose();
      return out;
    } else if (layers_[l].kernelType == SQUARE) { // non-homogeneous square
      const Double factor = static_cast<Double>(layers_[l].kernelParam);
      Array2d out = cosMap;
      out.colwise() *= normLMap;
      out.rowwise() *= normRMap.transpose();
      out = out.square();
      return out;
    } else {
      // LOG(ERROR) << "undefined kernel type";
      std::cerr << "undefined kernel type";
      return Array2d::Zero(1, 1);
    }
  }

  Array2d ukappa0(const Double* const cosines,
                const Double* const patches,
                const int32_t l) const {
    const size_t h = layerDims_[l].hconv;
    const size_t w = layerDims_[l].wconv;
    Eigen::Map<const Array2d> cosMap(cosines, h * w, h * w);
    Eigen::Map<const Array2d> patchMap(patches, h * w, h * w);
    if (layers_[l].kernelType == RELU) {
      // Array out = patch * (PI - cos.acos());
      Array2d cos = cosMap.min(1.0);
      Array2d out = cos.acos();
      out = patchMap * (static_cast<Double>(PI) - out);
      out /= PI;
      return out;
    } else {
      // LOG(ERROR) << "undefined kernel type";
      std::cerr << "undefined kernel type for NTK";
      return Array2d::Zero(1, 1);
    }
  }

  Tensor pool(Double* data, const int32_t l) const {
    const size_t h = layerDims_[l].hpool;
    const size_t w = layerDims_[l].wpool;
    const size_t hprev = layerDims_[l].hconv;
    const size_t wprev = layerDims_[l].wconv;

    const int sub = layers_[l].subsampling;
    const size_t sz = layerDims_[l].poolSize;

    Eigen::TensorMap<const Tensor> mapped(data, hprev, wprev, hprev, wprev);

    if (layers_[l].poolType == STRIDED) {
      Tensor out(h, w, h, w);
      FOR2DTO(i1, j1, h, w) {
        FOR2DTO(i2, j2, h, w) {
          out(i1, j1, i2, j2) = mapped(
              sub * (i1 + 1), sub * (j1 + 1), sub * (i2 + 1), sub * (j2 + 1));
        }
      }
      return out;
    }

    const auto& filt = poolFilter_[l];
    Eigen::Map<const Vector> filtMap(filt.data(), sz);

    Tensor pool1(hprev, wprev, hprev, w);
    FOR2DTO(i1, j1, hprev, wprev) {
      FOR2DTO(i2, j2, hprev, w) {
        if (sub * j2 + sz <= wprev) {
          Eigen::Map<const Vector> patch(&mapped(i1, j1, i2, sub * j2), sz);
          pool1(i1, j1, i2, j2) = filtMap.dot(patch);
        } else { // shorter
          const int ssz = static_cast<int>(wprev) - sub * j2;
          Eigen::Map<const Vector> shortFilt(filt.data(), ssz);
          Eigen::Map<const Vector> patch(&mapped(i1, j1, i2, sub * j2), ssz);
          pool1(i1, j1, i2, j2) = shortFilt.dot(patch);
        }
      }
    }
    Tensor pool2(hprev, wprev, h, w);
    FOR2DTO(i1, j1, hprev, wprev) {
      for (int i2 = 0; i2 < h; ++i2) {
        // pool2[i1,j1,i2,:]
        Eigen::Map<Vector> out(&pool2(i1, j1, i2, 0), w);
        if (sub * i2 + sz <= hprev) {
          // pool1[i1,j1,sub*i2:(sub*i2+sz),:]
          Eigen::Map<const Matrix> patch(&pool1(i1, j1, sub * i2, 0), sz, w);
          out = patch.transpose() * filtMap;
        } else { // shorter
          const int ssz = static_cast<int>(hprev) - sub * i2;
          Eigen::Map<const Vector> shortFilt(filt.data(), ssz);
          // pool1[i1,j1,sub*i2:(sub*i2+ssz),:]
          Eigen::Map<const Matrix> patch(&pool1(i1, j1, sub * i2, 0), ssz, w);
          out = patch.transpose() * shortFilt;
        }
      }
    }
    Tensor pool3(hprev, w, h, w);
    FOR2DTO(i1, j1, hprev, w) {
      // pool3[i1,j1,:,:]
      Eigen::Map<Vector> out(&pool3(i1, j1, 0, 0), h * w);
      if (sub * j1 + sz <= wprev) {
        // pool2[i1,sub*j1:(sub*j1+sz),:,:]
        Eigen::Map<const Matrix> patch(&pool2(i1, sub * j1, 0, 0), sz, h * w);
        out = patch.transpose() * filtMap;
      } else { // shorter
        const int ssz = static_cast<int>(wprev) - sub * j1;
        Eigen::Map<const Vector> shortFilt(filt.data(), ssz);
        // pool2[i1,sub*j1:(sub*j1+ssz),:,:]
        Eigen::Map<const Matrix> patch(&pool2(i1, sub * j1, 0, 0), ssz, h * w);
        out = patch.transpose() * shortFilt;
      }
    }
    Tensor poolout(h, w, h, w);
    for (int i1 = 0; i1 < h; ++i1) {
      // poolout[i1,:,:,:]
      Eigen::Map<Vector> out(&poolout(i1, 0, 0, 0), w * h * w);
      if (sub * i1 + sz <= hprev) {
        // pool3[sub*i1:(sub*i1+sz),:,:,:]
        Eigen::Map<const Matrix> patch(
            &pool3(sub * i1, 0, 0, 0), sz, w * h * w);
        out = patch.transpose() * filtMap;
      } else { // shorter
        const int ssz = static_cast<int>(hprev) - sub * i1;
        Eigen::Map<const Vector> shortFilt(filt.data(), ssz);
        // pool3[sub*i1:(sub*i1+ssz),:,:,:]
        Eigen::Map<const Matrix> patch(
            &pool3(sub * i1, 0, 0, 0), ssz, w * h * w);
        out = patch.transpose() * shortFilt;
      }
    }

    return poolout;
  }

  std::vector<Double> makeFilter(const size_t sz,
                                 const PoolType poolType = GAUSSIAN) const {
    if (poolType == GAUSSIAN) {
      const int sub = static_cast<int>(sz) / 2;
      const Double sigma = static_cast<Double>(sub) / std::sqrt(2.0);
      std::vector<Double> filt(sz);

      Double sum = 0.0;
      for (int i = -sub; i <= sub; ++i) {
        auto& f = filt[sub + i];
        f = std::exp(-(i * i) / (2 * sigma * sigma));
        sum += f;
      }
      for (int i = -sub; i <= sub; ++i) {
        filt[sub + i] /= sum;
      }
      return filt;
    } else if (poolType == AVERAGE) {
      std::vector<Double> filt(sz, 1. / sz);
      return filt;
    }
  }

  const std::vector<LayerParams> layers_;
  const size_t h_;
  const size_t w_;
  const size_t c_;
  const bool verbose_;

  Double cachedRF_;

  struct LayerDims {
    size_t hi;
    size_t wi;
    size_t hconv;
    size_t wconv;
    size_t hpool;
    size_t wpool;
    size_t poolSize;
    int interSize;
  };

  std::vector<LayerDims> layerDims_;
  std::vector<std::vector<Double>> poolFilter_;
};

template <typename Double>
Double computeAllKernel(const Double* const im1,
                        const Double* const im2,
                        const bool ntk,
                        const size_t h,
                        const size_t w,
                        const size_t c,
                        const std::vector<size_t>& patchSizes,
                        const std::vector<size_t>& subs,
                        const std::vector<size_t>& poolFactors,
                        const std::vector<int>& kernelTypes,
                        const std::vector<double>& kernelParams,
                        const std::vector<int>& pools,
                        const bool verbose = false) {
  std::vector<LayerParams> layers;
  const size_t L = patchSizes.size();
  for (size_t i = 0; i < L; ++i) {
    layers.push_back({patchSizes[i], subs[i], poolFactors[i],
                      static_cast<KernelType>(kernelTypes[i]), kernelParams[i],
                      /*zeroPad=*/true, static_cast<PoolType>(pools[i])});
  }
  std::vector<Double> norms1(L * h * w);
  std::vector<Double> norms2(L * h * w);
  std::vector<Double> normsInv1(L * h * w);
  std::vector<Double> normsInv2(L * h * w);

  CKNKernelMatrixEigen<Double> kernel(layers, h, w, c, verbose);
  kernel.computeNorms(im1, norms1.data(), normsInv1.data());
  kernel.computeNorms(im2, norms2.data(), normsInv2.data());
  return kernel.computeKernel(im1, im2, norms1.data(), norms2.data(),
                              normsInv1.data(), normsInv2.data(), ntk, verbose);
}

template <typename Double>
Double computeKernel(const Double* const im1,
                     const Double* const im2,
                     const Double* const norms1,
                     const Double* const norms2,
                     const Double* const normsInv1,
                     const Double* const normsInv2,
                     const bool ntk,
                     const size_t h,
                     const size_t w,
                     const size_t c,
                     const std::vector<size_t>& patchSizes,
                     const std::vector<size_t>& subs,
                     const std::vector<size_t>& poolFactors,
                     const std::vector<int>& kernelTypes,
                     const std::vector<double>& kernelParams,
                     const std::vector<int>& pools,
                     const bool verbose = false) {
  std::vector<LayerParams> layers;
  const size_t L = patchSizes.size();
  for (size_t i = 0; i < L; ++i) {
    layers.push_back({patchSizes[i], subs[i], poolFactors[i],
                      static_cast<KernelType>(kernelTypes[i]), kernelParams[i],
                      /*zeroPad=*/true, static_cast<PoolType>(pools[i])});
  }
  CKNKernelMatrixEigen<Double> kernel(layers, h, w, c, verbose);
  return kernel.computeKernel(im1, im2, norms1, norms2, normsInv1, normsInv2);
}

template <typename Double>
CKNKernelMatrixEigen<Double>* cknNew(const size_t h,
                                     const size_t w,
                                     const size_t c,
                                     const std::vector<size_t>& patchSizes,
                                     const std::vector<size_t>& subs,
                                     const std::vector<size_t>& poolFactors,
                                     const std::vector<int>& kernelTypes,
                                     const std::vector<double>& kernelParams,
                                     const std::vector<int>& pools,
                                     const bool verbose = false) {
  std::vector<LayerParams> layers;
  const size_t L = patchSizes.size();
  for (size_t i = 0; i < L; ++i) {
    layers.push_back({patchSizes[i], subs[i], poolFactors[i],
                      static_cast<KernelType>(kernelTypes[i]), kernelParams[i],
                      /*zeroPad=*/true, static_cast<PoolType>(pools[i])});
  }
  return new CKNKernelMatrixEigen<Double>(layers, h, w, c, verbose);
}

template <typename Double>
Double computeNorms(const Double* const im,
                    Double* norms,
                    Double* normsInv,
                    const size_t h,
                    const size_t w,
                    const size_t c,
                    const std::vector<size_t>& patchSizes,
                    const std::vector<size_t>& subs,
                    const std::vector<size_t>& poolFactors,
                    const std::vector<int>& kernelTypes,
                    const std::vector<double>& kernelParams,
                    const std::vector<int>& pools,
                    const bool verbose = false) {
  std::vector<LayerParams> layers;
  for (size_t i = 0; i < patchSizes.size(); ++i) {
    layers.push_back({patchSizes[i], subs[i], poolFactors[i],
                      static_cast<KernelType>(kernelTypes[i]), kernelParams[i],
                      /*zeroPad=*/true, static_cast<PoolType>(pools[i])});
  }

  CKNKernelMatrixEigen<Double> kernel(layers, h, w, c, verbose);
  kernel.computeNorms(im, norms, normsInv);
}
