#include <cassert>
#include <cmath>
// #include <glog/logging.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <map>


#define FOR4D(ii1, jj1, ii2, jj2)             \
  for (size_t ii1 = 0; ii1 < h; ++ii1)        \
    for (size_t jj1 = 0; jj1 < w; ++jj1)      \
      for (size_t ii2 = 0; ii2 < h; ++ii2)    \
        for (size_t jj2 = 0; jj2 < w; ++jj2)  \

#define FOR2D(ii, jj)                     \
  for (size_t ii = 0; ii < h; ++ii)        \
    for (size_t jj = 0; jj < w; ++jj)      \

#define FOR2DTO(ii, jj, hh, ww)                \
  for (size_t ii = 0; ii < hh; ++ii)        \
    for (size_t jj = 0; jj < ww; ++jj)      \

using sclock = std::chrono::system_clock;

namespace tuple_hash {

// Code from boost
// Reciprocal of the golden ratio helps spread entropy
//     and handles duplicates.
// See Mike Seymour in magic-numbers-in-boosthash-combine:
//     http://stackoverflow.com/questions/4948780

template <typename TT>
struct hash {
  size_t operator()(TT const& tt) const {
    return std::hash<TT>()(tt);
  }
};

namespace {
template <class T>
inline void hash_combine(std::size_t& seed, T const& v) {
  seed ^= hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Recursive template code derived from Matthieu M.
template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
  static void apply(size_t& seed, Tuple const& tuple) {
    HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
    hash_combine(seed, std::get<Index>(tuple));
  }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0> {
  static void apply(size_t& seed, Tuple const& tuple) {
    hash_combine(seed, std::get<0>(tuple));
  }
};
}

template <typename... TT>
struct hash<std::tuple<TT...>> {
  size_t operator()(std::tuple<TT...> const &tt) const {
    size_t seed = 0;
    HashValueImpl<std::tuple<TT...>>::apply(seed, tt);
    return seed;
  }
};
}

namespace {
static const bool useDP = true;
static const double PI = 3.141592653589793238463;

}

enum PoolType {
  GAUSSIAN = 0,
  STRIDED,
  AVERAGE
};

enum KernelType {
  EXP = 0,
  RELU
};

struct LayerParams {
  size_t patchSize;
  size_t subsampling;
  KernelType kernelType;
  double kernelParam; // sigma for EXP kernel
  bool zeroPad;
  PoolType poolType;
};

template <typename Double>
class CKNKernelMatrix {
 public:
  CKNKernelMatrix(const std::vector<LayerParams> &layers, const size_t h,
                  const size_t w, const size_t c, const bool verbose = false)
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
        // CHECK_GE(dims.hi, patch);
        assert(dims.hi >= patch);
        dims.hconv = dims.hi - patch + 1;
        dims.wconv = dims.wi - patch + 1;
      }
      const size_t sub = layers_[l].subsampling;
      if (layers_[l].poolType == GAUSSIAN) {
        dims.poolSize = 2 * sub + 1;
        poolFilter_[l] = makeFilter(dims.poolSize, layers_[l].poolType);
        // std::cout << "here ";
        // FOR2DTO(fi, fj, dims.poolSize, dims.poolSize) {
        //   std::cout << poolFilter_[l][fi * dims.poolSize + fj] << " ";
        // }
        // std::cout << std::endl;

        // sample at sub * i, with i = 1..hpool
        // CHECK_GE(dims.hconv, sub) << "too large subsampling";
        assert(dims.hconv >= sub);
        dims.hpool = dims.hconv / sub - 1;
        dims.wpool = dims.wconv / sub - 1;
      } else if (layers_[l].poolType == AVERAGE) {
        dims.poolSize = sub;
        poolFilter_[l] = makeFilter(dims.poolSize, layers_[l].poolType);
        dims.hpool = dims.hconv / sub;
        dims.wpool = dims.wconv / sub;
      } else if (layers_[l].poolType == STRIDED) {
        dims.poolSize = sub;
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

  Double computeKernel(const Double* im1, const Double* im2, const bool use_ntk = false) {
    std::vector<bool> ntks{false};
    if (use_ntk) {
      ntks.push_back(true);
    }

    auto startAlloc = sclock::now();
    pool_.resize(8);
    poolHalf_.resize(8);
    prod_.resize(8);
    conv_.resize(8);
    for (bool ntk : ntks) {
      for (OpType optype : {IM12, IM1, IM2}) {
        const size_t ntkim = ntkimkey(ntk, optype);
        pool_[ntkim].resize(h_ * w_ * h_ * w_);
        poolHalf_[ntkim].resize(h_ * w_ * h_ * w_);
        prod_[ntkim].resize(h_ * w_ * h_ * w_);
        conv_[ntkim].resize(h_ * w_ * h_ * w_);
      }
    }

    std::chrono::duration<double> elapsed = sclock::now() - startAlloc;
    std::cout << "alloc time: " << elapsed.count() << "\n";
    auto startCompute = sclock::now();

    size_t h = h_;
    size_t w = w_;

    // fill pool -1:
    auto start = sclock::now();
    for (bool ntk : ntks) {
      for (OpType optype : {IM12, IM1, IM2}) {
        FOR4D(i1, j1, i2, j2) {
          Double& val = pool(ntk, optype, /*l=*/-1, i1, j1, i2, j2);
          val = 0.0;
          for (size_t c = 0; c < c_; ++c) {
            if (optype == IM12) {
              val += im1[i1 * (w_ * c_) + j1 * c_ + c] *
                     im2[i2 * (w_ * c_) + j2 * c_ + c];
            } else if (optype == IM1) {
              val += im1[i1 * (w_ * c_) + j1 * c_ + c] *
                     im1[i2 * (w_ * c_) + j2 * c_ + c];
            } else {
              val += im2[i1 * (w_ * c_) + j1 * c_ + c] *
                     im2[i2 * (w_ * c_) + j2 * c_ + c];
            }
          }
        }
      }
    }
    elapsed = sclock::now() - start;
    std::cout << "init pool: " << elapsed.count() << " ";

    for (size_t l = 0; l < nlayers(); ++l) {
      start = sclock::now();
      //********** prod ************
      h = layerDims_[l].hconv;
      w = layerDims_[l].wconv;
      for (bool ntk : ntks) {
        for (OpType optype : {IM12, IM1, IM2}) {
          FOR4D(i1, j1, i2, j2) {
            Double& val = prod(ntk, optype, l, i1, j1, i2, j2);

            const int sz = layers_[l].patchSize;
            const int start = layers_[l].zeroPad ? -(sz - 1) / 2 : 0;
            const size_t hbelow = l >= 1 ? layerDims_[l-1].hpool : h_;
            const size_t wbelow = l >= 1 ? layerDims_[l-1].wpool : w_;
            val = 0.0;
            for (int i = start; i < start + sz; ++i) {
              for (int j = start; j < start + sz; ++j) {
                if (i1 + i >= 0 && i1 + i < hbelow && j1 + j >= 0 && j1 + j < wbelow
                    && i2 + i >= 0 && i2 + i < hbelow && j2 + j >= 0 && j2 + j < wbelow) {
                  val += pool(ntk, optype, l - 1, i1 + i, j1 + j, i2 + i, j2 + j);
                }
              }
            }
            // val /= (sz * sz);
          }
        }
      }
      elapsed = sclock::now() - start;
      std::cout << "prod: " << elapsed.count() << " ";


      start = sclock::now();
      //********** conv ************
      h = layerDims_[l].hconv;
      w = layerDims_[l].wconv;
      for (bool ntk : ntks) {
        for (OpType optype : {IM12, IM1, IM2}) {
          const OpType op1 = (optype == IM2) ? IM2 : IM1;
          const OpType op2 = (optype == IM1) ? IM1 : IM2;
          FOR2D(i1, j1) {
            const Double sqnorm1 = prod(/*ntk=*/false, op1, l, i1, j1, i1, j1);
            FOR2D(i2, j2) {
              Double& val = conv(ntk, optype, l, i1, j1, i2, j2);
              const Double normProd =
                  std::sqrt(sqnorm1 * prod(/*ntk=*/false, op2, l, i2, j2, i2, j2));
              Double cosine = 0.0;
              if (normProd > 1e-8) {
                cosine =
                    prod(/*ntk=*/false, optype, l, i1, j1, i2, j2) / normProd;
              }
              val = normProd * kappa(cosine, /*derivative=*/false,
                                     layers_[l].kernelType,
                                     layers_[l].kernelParam);

              if (ntk) {
                // add second term based on tensor product
                Double dotprod_ntk =
                    prod(/*ntk=*/true, optype, l, i1, j1, i2, j2);

                val += dotprod_ntk * kappa(cosine, /*derivative=*/true,
                                           layers_[l].kernelType,
                                           layers_[l].kernelParam);
              }
            }
          }
        }
      }
      elapsed = sclock::now() - start;
      std::cout << "conv: " << elapsed.count() << " ";

      // std::cout << pool(false, IM12, l - 1, 0, 1, 2, 3) << " "
      //           << prod(false, IM12, l, 0, 1, 2, 3) << " "
      //           << conv(false, IM12, l, 0, 1, 2, 3) << " \n";

      // treat last pooling layer separately if cheaper
      // if (l == nlayers() - 1) {
      //   break;
      // }

      start = sclock::now();
      //********** pool ************
      h = layerDims_[l].hpool;
      w = layerDims_[l].wpool;
      const int sub = layers_[l].subsampling;
      for (bool ntk : ntks) {
        for (OpType optype : {IM12, IM1, IM2}) {
          if (layers_[l].poolType == STRIDED) {
            FOR4D(i1, j1, i2, j2) {
              if (l == nlayers() - 1 && (i1 != i2 || j1 != j2)) {
                continue;
              }
              Double& val = pool(ntk, optype, l, i1, j1, i2, j2);
              val = conv(ntk, optype, l, sub * (i1 + 1), sub * (j1 + 1),
                         sub * (i2 + 1), sub * (j2 + 1));
            }
          } else if (true) { // AVERAGE, GAUSSIAN
            const auto& filt = poolFilter_[l];
            const size_t sub = layers_[l].subsampling;
            const size_t sz = layerDims_[l].poolSize;

            // separate 4D convolution in two 2D convolutions
            FOR2DTO(i1, j1, layerDims_[l].hconv, layerDims_[l].wconv) {
              FOR2D(i2, j2) {
                Double& val = poolHalf(ntk, optype, l, i1, j1, i2, j2);

                val = 0.0;
                FOR2DTO(fi, fj, sz, sz) {
                  if (sub * i2 + fi >= layerDims_[l].hconv ||
                      sub * j2 + fj >= layerDims_[l].wconv) {
                    continue;
                  }
                  val +=
                      filt[fi * sz + fj] * conv(ntk, optype, l, i1, j1,
                                                sub * i2 + fi, sub * j2 + fj);
                }
              }
            }
            FOR2D(i1, j1) {
              FOR2DTO(fi, fj, sz, sz) {
                const Double fval = filt[fi * sz + fj];
                FOR2D(i2, j2) {
                  if (l == nlayers() - 1 && (i1 != i2 || j1 != j2)) {
                    continue;
                  }
                  Double& val = pool(ntk, optype, l, i1, j1, i2, j2);
                  if (fi == 0 && fj == 0) {
                    val = 0.0;
                  }
                  if (sub * i1 + fi >= layerDims_[l].hconv ||
                      sub * j1 + fj >= layerDims_[l].wconv) {
                    continue;
                  }
                  val += fval * poolHalf(ntk, optype, l, sub * i1 + fi,
                                         sub * j1 + fj, i2, j2);
                }
              }
            }
          } else { // TODO: remove (slower, 4D conv)
            const auto& filt = poolFilter_[l];
            const int sub = layers_[l].subsampling;
            const int sz = layerDims_[l].poolSize;
            assert(sz == 2 * sub + 1);

            FOR4D(i1, j1, i2, j2) {
              Double& val = pool(ntk, optype, l, i1, j1, i2, j2);
              val = 0.0;
              FOR2DTO(fi, fj, sz - 1, sz - 1) {
                FOR2DTO(fii, fjj, sz - 1, sz - 1) {
                  const Double fval = filt[fi * sz + fj] * filt[fii * sz + fjj];
                  val +=
                      fval * conv(ntk, optype, l, sub * i1 + fi, sub * j1 + fj,
                                  sub * i2 + fii, sub * j2 + fjj);
                }
              }
            }
          }
        }
      }
      elapsed = sclock::now() - start;
      std::cout << "pool: " << elapsed.count() << " ";
    }

    Double ret = 0.0;
    h = layerDims_[nlayers() - 1].hpool;
    w = layerDims_[nlayers() - 1].wpool;
    FOR2D(i, j) {
      ret += pool(use_ntk, IM12, nlayers() - 1, i, j, i, j);
    }

    elapsed = sclock::now() - startCompute;
    std::cout << "compute time: " << elapsed.count() << "\n";
    return ret;
  }

private:
  enum OpType { IM12 = 0, IM1, IM2 };

  size_t ntkimkey(const bool ntk, const OpType optype) {
    return (static_cast<size_t>(ntk) << 2) + static_cast<size_t>(optype);
  }

  Double& pool(const bool ntk,
               const OpType optype,
               const int32_t l,
               const size_t i1,
               const size_t j1,
               const size_t i2,
               const size_t j2) {
    const size_t ntkim = ntkimkey(ntk, optype);
    size_t h, w;
    if (l == -1) {
      h = h_;
      w = w_;
    } else {
      h = layerDims_[l].hpool;
      w = layerDims_[l].wpool;
    }
    size_t idx = i1 * w * h * w + j1 * h * w + i2 * w + j2;
    return pool_[ntkim][idx];
  }

  Double& poolHalf(const bool ntk,
                   const OpType optype,
                   const int32_t l,
                   const size_t i1,
                   const size_t j1,
                   const size_t i2,
                   const size_t j2) {
    const size_t ntkim = ntkimkey(ntk, optype);
    const size_t hc = layerDims_[l].hconv;
    const size_t wc = layerDims_[l].wconv;
    const size_t h = layerDims_[l].hpool;
    const size_t w = layerDims_[l].wpool;
    size_t idx = i1 * wc * h * w + j1 * h * w + i2 * w + j2;
    return poolHalf_[ntkim][idx];
  }

  Double& prod(const bool ntk,
               const OpType optype,
               const int32_t l,
               const size_t i1,
               const size_t j1,
               const size_t i2,
               const size_t j2) {
    const size_t ntkim = ntkimkey(ntk, optype);
    const size_t h = layerDims_[l].hconv;
    const size_t w = layerDims_[l].wconv;
    size_t idx = i1 * w * h * w + j1 * h * w + i2 * w + j2;
    return prod_[ntkim][idx];
  }

  Double& conv(const bool ntk,
               const OpType optype,
               const int32_t l,
               const size_t i1,
               const size_t j1,
               const size_t i2,
               const size_t j2) {
    const size_t ntkim = ntkimkey(ntk, optype);
    const size_t h = layerDims_[l].hconv;
    const size_t w = layerDims_[l].wconv;
    size_t idx = i1 * w * h * w + j1 * h * w + i2 * w + j2;
    return conv_[ntkim][idx];
  }

  std::vector<Double> makeFilter(const size_t sz,
                                 const PoolType poolType = GAUSSIAN) const {
    if (poolType == GAUSSIAN) {
      const int sub = static_cast<int>(sz) / 2;
      const Double sigma = static_cast<Double>(sub) / std::sqrt(2.0);
      std::vector<Double> filt(sz * sz);

      Double sum = 0.0;
      for (int i = -sub; i <= sub; ++i) {
        for (int j = -sub; j <= sub; ++j) {
          auto& f = filt[(sub + i) * sz + (sub + j)];
          if (poolType == GAUSSIAN) {
            f = std::exp(-(i * i + j * j) / (2 * sigma * sigma));
          } else if (poolType == AVERAGE) {
            // f = (std::abs(i) <= 1 && std::abs(j) <= 1) ? 1.0 : 0.0;
            if (i >= -1 && i <= 0 && j >= -1 && j <= 0) {
              f = 1.0;
            } else {
              f = 0.0;
            }
          } else if (poolType == STRIDED) {
            f = (i == 0 && j == 0) ? 1.0 : 0.0;
          } else {
            // LOG(FATAL) << "bad pool type";
            std::cerr << "bad pool type";
          }
          sum += f;
        }
      }
      for (int i = -sub; i <= sub; ++i) {
        for (int j = -sub; j <= sub; ++j) {
          filt[(sub + i) * sz + (sub + j)] /= sum;
        }
      }
      return filt;
    } else if (poolType == AVERAGE) {
      std::vector<Double> filt(sz * sz, 1. / (sz * sz));
      return filt;
    }
  }

  Double kappa(const Double cosine,
               const bool derivative = false,
               const KernelType kernelType = EXP,
               const Double kernelParam = 1.0) const {
    if (kernelType == EXP) {
      const auto sigma = kernelParam;
      return std::exp((cosine - 1) / (sigma * sigma));
    } else if (kernelType == RELU && !derivative) {
      if (cosine > 0.9999) {
        return 1;
      } else {
        return (cosine * (PI - std::acos(cosine)) + std::sqrt(1. - cosine * cosine)) / PI;
      }
    } else if (kernelType == RELU && derivative) {
      if (cosine > 0.9999) {
        return 1;
      } else {
        return 1. - std::acos(cosine) / PI;
      }
    } else {
      // LOG(ERROR) << "undefined kernel type";
      std::cerr << "undefined kernel type";
      return 0.;
    }
  }

  const std::vector<LayerParams> layers_;
  const size_t h_;
  const size_t w_;
  const size_t c_;
  const bool verbose_;

  struct LayerDims {
    size_t hi;
    size_t wi;
    size_t hconv;
    size_t wconv;
    size_t hpool;
    size_t wpool;
    size_t poolSize;
  };

  std::vector<LayerDims> layerDims_;
  std::vector<std::vector<Double>> poolFilter_;

  std::vector<std::vector<Double>> pool_;
  std::vector<std::vector<Double>> poolHalf_;
  std::vector<std::vector<Double>> prod_;
  std::vector<std::vector<Double>> conv_;

};


template <typename Double>
class CKNKernelMatrixLazy {
 public:
  CKNKernelMatrixLazy(const std::vector<LayerParams> &layers, const size_t h,
                  const size_t w, const size_t c, const bool verbose = false)
      : layers_(layers), h_(h), w_(w), c_(c), verbose_(verbose) {
    // CHECK(h == w) << "only square images are supported for now";
    assert(h == w);
    layerDims_.resize(layers.size());
    layerDims_[0].hi = h;
    layerDims_[0].wi = w;

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
        // CHECK_GE(dims.hi, patch);
        assert(dims.hi >= patch);
        dims.hconv = dims.hi - patch + 1;
        dims.wconv = dims.wi - patch + 1;
      }
      const size_t sub = layers_[l].subsampling;
      dims.poolSize = 2 * sub + 1;
      poolFilter_.emplace_back(makeFilter(dims.poolSize, layers_[l].poolType));

      // sample at sub * i, with i = 1..hpool
      // CHECK_GE(dims.hconv, sub) << "too large subsampling";
      assert(dims.hconv >= sub);
      dims.hpool = dims.hconv / sub - 1;
      dims.wpool = dims.wconv / sub - 1;
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

  Double computeKernel(const Double* im1, const Double* im2, const bool ntk = false) {
    poolMap_.clear();
    convMap_.clear();
    prodMap_.clear();

    std::vector<const Double*> ims{im1, im2};
    // for (int l = 0; l < nlayers(); ++l) {
    //   std::cout << pool(ims, false, 0, 1, l - 1, 0, 1, 2, 3) << " "
    //             << prod(ims, false, 0, 1, l, 0, 1, 2, 3) << " "
    //             << conv(ims, false, 0, 1, l, 0, 1, 2, 3) << " \n";
    // }
    Double val = 0.0;
    for (size_t i = 0; i < layerDims_[nlayers() - 1].hpool; ++i) {
      for (size_t j = 0; j < layerDims_[nlayers() - 1].wpool; ++j) {
        val += pool(ims, ntk, 0, 1, nlayers() - 1, i, j, i, j);
      }
    }
    return val;
  }

private:
  std::vector<Double> makeFilter(const size_t sz,
                                 const PoolType poolType = GAUSSIAN) const {
    const int sub = static_cast<int>(sz) / 2;
    const Double sigma = static_cast<Double>(sub) / std::sqrt(2.0);
    std::vector<Double> filt(sz * sz);

    Double sum = 0.0;
    for (int i = -sub; i <= sub; ++i) {
      for (int j = -sub; j <= sub; ++j) {
        auto& f = filt[(sub + i) * sz + (sub + j)];
        if (poolType == GAUSSIAN) {
          f = std::exp(-(i * i + j * j) / (2 * sigma * sigma));
        } else if (poolType == AVERAGE) {
          // f = (std::abs(i) <= 1 && std::abs(j) <= 1) ? 1.0 : 0.0;
          if (i >= -1 && i <= 0 && j >= -1 && j <= 0) {
            f = 1.0;
          } else {
            f = 0.0;
          }
        } else if (poolType == STRIDED) {
          f = (i == 0 && j == 0) ? 1.0 : 0.0;
        } else {
          // LOG(FATAL) << "bad pool type";
          std::cerr << "bad pool type";
        }
        sum += f;
      }
    }
    for (int i = -sub; i <= sub; ++i) {
      for (int j = -sub; j <= sub; ++j) {
        filt[(sub + i) * sz + (sub + j)] /= sum;
      }
    }

    return filt;
  }

  Double pool(const std::vector<const Double *> &ims, const bool ntk, const uint8_t im1Idx,
              const uint8_t im2Idx, const int32_t l, const size_t i1,
              const size_t j1, const size_t i2, const size_t j2) {
    /* CHECK_GE(l, -1);
    if (l >= 0) {
      CHECK_GE(i1, 0);
      CHECK_LT(i1, layerDims_[l].hpool);
      CHECK_GE(j1, 0);
      CHECK_LT(j1, layerDims_[l].wpool);
      CHECK_GE(i2, 0);
      CHECK_LT(i2, layerDims_[l].hpool);
      CHECK_GE(j2, 0);
      CHECK_LT(j2, layerDims_[l].wpool);
    } else {
      CHECK_GE(i1, 0);
      CHECK_LT(i1, h_);
      CHECK_GE(j1, 0);
      CHECK_LT(j1, w_);
      CHECK_GE(i2, 0);
      CHECK_LT(i2, h_);
      CHECK_GE(j2, 0);
      CHECK_LT(j2, w_);
    } */

    auto key = KeyT(ntk, im1Idx, im2Idx, l, i1, j1, i2, j2);
    auto pos = poolMap_.find(key);
    if (pos != poolMap_.end()) {
      return pos->second;
    }
    const Double* const im1 = ims[im1Idx];
    const Double* const im2 = ims[im2Idx];

    if (l == -1) { // image pixels
      Double val = 0.0;
      for (size_t c = 0; c < c_; ++c) {
        val += im1[i1 * (w_ * c_) + j1 * c_ + c] * im2[i2 * (w_ * c_) + j2 * c_ + c];
      }
      if (useDP) {
        return poolMap_[key] = val;
      } else {
        return val;
      }
    }

    const int sub = layers_[l].subsampling;
    const int sz = 2 * sub + 1; // h or w of pool filter
    const auto& filt = poolFilter_[l];
    // center coords in conv layer
    const size_t iconv1 = sub * (i1 + 1);
    const size_t jconv1 = sub * (j1 + 1);
    const size_t iconv2 = sub * (i2 + 1);
    const size_t jconv2 = sub * (j2 + 1);
    Double val = 0.0;
    for (int ii1 = -sub; ii1 <= sub; ++ii1) {
      for (int ii2 = -sub; ii2 <= sub; ++ii2) {
        for (int jj1 = -sub; jj1 <= sub; ++jj1) {
          for (int jj2 = -sub; jj2 <= sub; ++jj2) {
            // LOG_EVERY_N(INFO, 10000) << filt.size() << " "
            //     << sub << " " << sz << " " << ii1 << " " << jj1
            //     << " " << ii2 << " " << jj2;
            if (iconv1 + ii1 >= layerDims_[l].hconv ||
                jconv1 + jj1 >= layerDims_[l].wconv ||
                iconv2 + ii2 >= layerDims_[l].hconv ||
                jconv2 + jj2 >= layerDims_[l].wconv) {
              continue;
            }
            const Double fval = filt[(sub + ii1) * sz + (sub + jj1)] *
                                filt[(sub + ii2) * sz + (sub + jj2)];
            if (fval > 1e-9) {
              val += fval * conv(ims, ntk, im1Idx, im2Idx, l, iconv1 + ii1,
                                 jconv1 + jj1, iconv2 + ii2, jconv2 + jj2);
            }
          }
        }
      }
    }

    if (pos != poolMap_.end() && std::abs(pos->second - val) > 1e-5) {
      // LOG_EVERY_N(ERROR, 10000) << pos->second << " vs " << val << ": " << im1Idx << im2Idx << l;
      std::cerr << pos->second << " vs " << val << ": " << im1Idx << im2Idx << l;
    }
    if (useDP) {
      return poolMap_[key] = val;
    } else {
      return val;
    }
  }

  // send patch to RKHS
  Double conv(const std::vector<const Double *> &ims, const bool ntk, const uint8_t im1Idx,
              const uint8_t im2Idx, const int32_t l, const size_t i1,
              const size_t j1, const size_t i2, const size_t j2) {
    /* CHECK_GE(l, 0);
    CHECK_GE(i1, 0); CHECK_LT(i1, layerDims_[l].hconv);
    CHECK_GE(j1, 0); CHECK_LT(j1, layerDims_[l].wconv);
    CHECK_GE(i2, 0); CHECK_LT(i2, layerDims_[l].hconv);
    CHECK_GE(j2, 0); CHECK_LT(j2, layerDims_[l].wconv);
    */
    auto key = KeyT(ntk, im1Idx, im2Idx, l, i1, j1, i2, j2);
    auto pos = convMap_.find(key);
    if (pos != convMap_.end()) {
      return pos->second;
    }

    Double norm1 = std::sqrt(prod(ims, /*ntk=*/false, im1Idx, im1Idx, l, i1, j1, i1, j1));
    Double norm2 = std::sqrt(prod(ims, /*ntk=*/false, im2Idx, im2Idx, l, i2, j2, i2, j2));

    Double cosine = 0.0;
    if (norm1 > 1e-6 && norm2 > 1e-6) {
      cosine = prod(ims, /*ntk=*/false, im1Idx, im2Idx, l, i1, j1, i2, j2) / norm1 / norm2;
    }
    Double val = norm1 * norm2 *
                 kappa(cosine, /*derivative=*/false, layers_[l].kernelType,
                       layers_[l].kernelParam);

    if (ntk) {
      // add second term based on tensor product
      Double dotprod_ntk =
          prod(ims, /*ntk=*/true, im1Idx, im2Idx, l, i1, j1, i2, j2);

      val += dotprod_ntk * kappa(cosine, /*derivative=*/true,
                                 layers_[l].kernelType, layers_[l].kernelParam);
    }

    if (pos != convMap_.end() && std::abs(pos->second - val) > 1e-5) {
      // LOG_EVERY_N(ERROR, 10000) << pos->second << " vs " << val << ": " << im1Idx << im2Idx << l;
      std::cerr << pos->second << " vs " << val << ": " << im1Idx << im2Idx << l;
    }
    if (useDP) {
      return convMap_[key] = val;
    } else {
      return val;
    }
  }

  // inner product on patch
  Double prod(const std::vector<const Double *> &ims, const bool ntk, const uint8_t im1Idx,
              const uint8_t im2Idx, const int32_t l, const size_t i1,
              const size_t j1, const size_t i2, const size_t j2) {
    /* CHECK_GE(l, 0);
    CHECK_GE(i1, 0); CHECK_LT(i1, layerDims_[l].hi);
    CHECK_GE(j1, 0); CHECK_LT(j1, layerDims_[l].wi);
    CHECK_GE(i2, 0); CHECK_LT(i2, layerDims_[l].hi);
    CHECK_GE(j2, 0); CHECK_LT(j2, layerDims_[l].wi);
    */
    auto key = KeyT(ntk, im1Idx, im2Idx, l, i1, j1, i2, j2);
    auto pos = prodMap_.find(key);
    if (pos != prodMap_.end()) {
      return pos->second;
    }

    const int sz = layers_[l].patchSize;
    const int start = layers_[l].zeroPad ? -(sz - 1) / 2 : 0;
    const size_t hbelow = l >= 1 ? layerDims_[l-1].hpool : h_;
    const size_t wbelow = l >= 1 ? layerDims_[l-1].wpool : w_;
    Double val = 0.0;
    for (int i = start; i < start + sz; ++i) {
      for (int j = start; j < start + sz; ++j) {
        if (i1 + i >= 0 && i1 + i < hbelow && j1 + j >= 0 && j1 + j < wbelow
            && i2 + i >= 0 && i2 + i < hbelow && j2 + j >= 0 && j2 + j < wbelow) {
          val += pool(ims, ntk, im1Idx, im2Idx, l - 1, i1 + i, j1 + j, i2 + i, j2 + j);
        }
      }
    }
    // val /= (sz * sz);

    if (pos != prodMap_.end() && std::abs(pos->second - val) > 1e-5) {
      // LOG_EVERY_N(ERROR, 10000) << pos->second << " vs " << val << ": " << im1Idx << im2Idx << l;
      std::cerr << pos->second << " vs " << val << ": " << im1Idx << im2Idx << l;
    }
    if (useDP) {
      return prodMap_[key] = val;
    } else {
      return val;
    }
  }

  Double kappa(const Double cosine,
               const bool derivative = false,
               const KernelType kernelType = EXP,
               const Double kernelParam = 1.0) const {
    if (kernelType == EXP) {
      const auto sigma = kernelParam;
      return std::exp((cosine - 1) / (sigma * sigma));
    } else if (kernelType == RELU && !derivative) {
      if (cosine > 0.9999) {
        return 1;
      } else {
        return (cosine * (PI - std::acos(cosine)) + std::sqrt(1. - cosine * cosine)) / PI;
      }
    } else if (kernelType == RELU && derivative) {
      if (cosine > 0.9999) {
        return 1;
      } else {
        return 1. - std::acos(cosine) / PI;
      }
    } else {
      // LOG(ERROR) << "undefined kernel type";
      std::cerr << "undefined kernel type";
      return 0.;
    }
  }

  const std::vector<LayerParams> layers_;
  const size_t h_;
  const size_t w_;
  const size_t c_;
  const bool verbose_;

  struct LayerDims {
    size_t hi;
    size_t wi;
    size_t hconv;
    size_t wconv;
    size_t hpool;
    size_t wpool;
    size_t poolSize;
  };

  std::vector<LayerDims> layerDims_;
  std::vector<std::vector<Double>> poolFilter_;

  using KeyT =
      std::tuple<bool, uint8_t, uint8_t, int32_t, size_t, size_t, size_t, size_t>;
  using MapT = std::unordered_map<KeyT, Double, tuple_hash::hash<KeyT>>;
  // using MapT = std::map<KeyT, Double>;

  MapT poolMap_;
  MapT convMap_;
  MapT prodMap_;
};

template <typename Double>
Double computeKernel(const Double *const im1, const Double *const im2,
                     const bool ntk,
                     const size_t h, const size_t w, const size_t c,
                     const std::vector<size_t>& patchSizes,
                     const std::vector<size_t>& subs,
                     const std::vector<int>& kernelTypes,
                     const std::vector<double>& kernelParams,
                     const std::vector<int>& pools,
                     const bool verbose = false,
                     const bool lazy = false) {
  std::vector<LayerParams> layers;
  for (size_t i = 0; i < patchSizes.size(); ++i) {
    layers.push_back({patchSizes[i], subs[i], static_cast<KernelType>(kernelTypes[i]),
                      kernelParams[i], /*zeroPad=*/true, static_cast<PoolType>(pools[i])});
  }

  if (lazy) {
    CKNKernelMatrixLazy<Double> kernelLazy(layers, h, w, c, verbose);
    return kernelLazy.computeKernel(im1, im2, ntk);
  } else {
    CKNKernelMatrix<Double> kernel(layers, h, w, c, verbose);
    return kernel.computeKernel(im1, im2, ntk);
  }
}

template <typename Double>
Double computeKernelLazy(const Double *const im1, const Double *const im2,
                     const bool ntk,
                     const size_t h, const size_t w, const size_t c,
                     const std::vector<size_t>& patchSizes,
                     const std::vector<size_t>& subs,
                     const std::vector<int>& kernelTypes,
                     const std::vector<double>& kernelParams,
                     const std::vector<int>& pools,
                     const bool verbose = false) {
  std::vector<LayerParams> layers;
  for (size_t i = 0; i < patchSizes.size(); ++i) {
    layers.push_back({patchSizes[i], subs[i], static_cast<KernelType>(kernelTypes[i]),
                      kernelParams[i], /*zeroPad=*/true, static_cast<PoolType>(pools[i])});
  }

  CKNKernelMatrixLazy<Double> kernel(layers, h, w, c, verbose);
  return kernel.computeKernel(im1, im2, ntk);
}
