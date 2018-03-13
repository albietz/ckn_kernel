#include <cmath>
#include <glog/logging.h>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <map>

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
}

enum PoolType {
  GAUSSIAN = 0,
  STRIDED,
  AVERAGE
};

struct LayerParams {
  size_t patchSize;
  size_t subsampling;
  double sigma;
  bool zeroPad;
  PoolType poolType;
};

template <typename Double>
class CKNKernelMatrix {
 public:
  CKNKernelMatrix(const std::vector<LayerParams> &layers, const size_t h,
                  const size_t w, const size_t c)
      : layers_(layers), h_(h), w_(w), c_(c) {
    CHECK(h == w) << "only square images are supported for now";
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
        CHECK_GE(dims.hi, patch);
        dims.hconv = dims.hi - patch + 1;
        dims.wconv = dims.wi - patch + 1;
      }
      const size_t sub = layers_[l].subsampling;
      dims.poolSize = 2 * sub + 1;
      poolFilter_.emplace_back(makeFilter(dims.poolSize, layers_[l].poolType));

      // sample at sub * i, with i = 1..hpool
      CHECK_GE(dims.hconv, sub) << "too large subsampling";
      dims.hpool = dims.hconv / sub - 1;
      dims.wpool = dims.wconv / sub - 1;
      LOG(INFO) << "layer " << l << "(" << patch << "," << sub
                << "): " << dims.hi << "x" << dims.wi << " -> " << dims.hconv
                << "x" << dims.wconv << " -> " << dims.hpool << "x"
                << dims.wpool;
    }
  }

  size_t nlayers() const {
    return layers_.size();
  }

  Double computeKernel(const Double* im1, const Double* im2) {
    poolMap_.clear();
    convMap_.clear();
    prodMap_.clear();

    std::vector<const Double*> ims{im1, im2};
    Double val = 0.0;
    for (size_t i = 0; i < layerDims_[nlayers() - 1].hpool; ++i) {
      for (size_t j = 0; j < layerDims_[nlayers() - 1].wpool; ++j) {
        val += pool(ims, 0, 1, nlayers() - 1, i, j, i, j);
      }
    }
    return val;
  }

private:
  // whether to compute across images or on a single image
  // used for the dynamic programming maps
  enum OpType { IM12, IM1, IM2 };

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
          f = 1.0;
        } else if (poolType == STRIDED) {
          f = (i == 0 && j == 0) ? 1.0 : 0.0;
        } else {
          LOG(FATAL) << "bad pool type";
        }
        sum += f;
      }
    }
    for (int i = -sub; i < sub; ++i) {
      for (int j = -sub; j < sub; ++j) {
        filt[(sub + i) * sz + (sub + j)] /= sum;
      }
    }

    return filt;
  }

  Double pool(const std::vector<const Double *> &ims, const uint8_t im1Idx,
              const uint8_t im2Idx, const int32_t l, const size_t i1,
              const size_t j1, const size_t i2, const size_t j2) {
    CHECK_GE(l, -1);
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
    }

    auto key = KeyT(im1Idx, im2Idx, l, i1, j1, i2, j2);
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
    for (int ii1 = -sub; ii1 < sub; ++ii1) {
      for (int ii2 = -sub; ii2 < sub; ++ii2) {
        for (int jj1 = -sub; jj1 < sub; ++jj1) {
          for (int jj2 = -sub; jj2 < sub; ++jj2) {
            // LOG_EVERY_N(INFO, 10000) << filt.size() << " "
            //     << sub << " " << sz << " " << ii1 << " " << jj1
            //     << " " << ii2 << " " << jj2;
            val += filt[(sub + ii1) * sz + (sub + jj1)] *
                   filt[(sub + ii2) * sz + (sub + jj2)] *
                   conv(ims, im1Idx, im2Idx, l, iconv1 + ii1, jconv1 + jj1,
                        iconv2 + ii2, jconv2 + jj2);
          }
        }
      }
    }

    if (pos != poolMap_.end() && std::abs(pos->second - val) > 1e-5) {
      LOG_EVERY_N(ERROR, 10000) << pos->second << " vs " << val << ": " << im1Idx << im2Idx << l;
    }
    if (useDP) {
      return poolMap_[key] = val;
    } else {
      return val;
    }
  }

  // send patch to RKHS
  Double conv(const std::vector<const Double *> &ims, const uint8_t im1Idx,
              const uint8_t im2Idx, const int32_t l, const size_t i1,
              const size_t j1, const size_t i2, const size_t j2) {
    CHECK_GE(l, 0);
    CHECK_GE(i1, 0); CHECK_LT(i1, layerDims_[l].hconv);
    CHECK_GE(j1, 0); CHECK_LT(j1, layerDims_[l].wconv);
    CHECK_GE(i2, 0); CHECK_LT(i2, layerDims_[l].hconv);
    CHECK_GE(j2, 0); CHECK_LT(j2, layerDims_[l].wconv);
    auto key = KeyT(im1Idx, im2Idx, l, i1, j1, i2, j2);
    auto pos = convMap_.find(key);
    if (pos != convMap_.end()) {
      return pos->second;
    }

    Double norm1 = std::sqrt(prod(ims, im1Idx, im1Idx, l, i1, j1, i1, j1));
    Double norm2 = std::sqrt(prod(ims, im2Idx, im2Idx, l, i2, j2, i2, j2));

    Double cosine = 0.0;
    if (norm1 > 1e-6 && norm2 > 1e-6) {
      cosine = prod(ims, im1Idx, im2Idx, l, i1, j1, i2, j2) / norm1 / norm2;
    }
    Double val = norm1 * norm2 * kappa(cosine, layers_[l].sigma);

    if (pos != convMap_.end() && std::abs(pos->second - val) > 1e-5) {
      LOG_EVERY_N(ERROR, 10000) << pos->second << " vs " << val << ": " << im1Idx << im2Idx << l;
    }
    if (useDP) {
      return convMap_[key] = val;
    } else {
      return val;
    }
  }

  // inner product on patch
  Double prod(const std::vector<const Double *> &ims, const uint8_t im1Idx,
              const uint8_t im2Idx, const int32_t l, const size_t i1,
              const size_t j1, const size_t i2, const size_t j2) {
    CHECK_GE(l, 0);
    CHECK_GE(i1, 0); CHECK_LT(i1, layerDims_[l].hi);
    CHECK_GE(j1, 0); CHECK_LT(j1, layerDims_[l].wi);
    CHECK_GE(i2, 0); CHECK_LT(i2, layerDims_[l].hi);
    CHECK_GE(j2, 0); CHECK_LT(j2, layerDims_[l].wi);
    auto key = KeyT(im1Idx, im2Idx, l, i1, j1, i2, j2);
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
          val += pool(ims, im1Idx, im2Idx, l - 1, i1 + i, j1 + j, i2 + i, j2 + j);
        }
      }
    }
    val /= (sz * sz);

    if (pos != prodMap_.end() && std::abs(pos->second - val) > 1e-5) {
      LOG_EVERY_N(ERROR, 10000) << pos->second << " vs " << val << ": " << im1Idx << im2Idx << l;
    }
    if (useDP) {
      return prodMap_[key] = val;
    } else {
      return val;
    }
  }

  Double kappa(const Double cosine, const Double sigma) const {
    return std::exp((cosine - 1) / (sigma * sigma));
  }

  uint64_t makeKey(const uint8_t im1Idx, const uint8_t im2Idx, const int32_t l,
                   const size_t i1, const size_t j1, const size_t i2,
                   const size_t j2) const {
    uint64_t k = (static_cast<uint64_t>(im1Idx) << 62) +
                 (static_cast<uint64_t>(im2Idx) << 60) +
                 (static_cast<uint64_t>(l) << 48) + (i1 << 36) + (j1 << 24) +
                 (i2 << 12) + j2;
    // LOG_EVERY_N(INFO, 1) << k << " = " << type << " " << l << " " << i1 << " " << j1 << " " << i2 << " " << j2;
    return k;
  }

  const std::vector<LayerParams> layers_;
  const size_t h_;
  const size_t w_;
  const size_t c_;

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
      std::tuple<uint8_t, uint8_t, int32_t, size_t, size_t, size_t, size_t>;
  // using MapT = std::unordered_map<KeyT, Double, tuple_hash::hash<KeyT>>;
  using MapT = std::map<KeyT, Double>;

  MapT poolMap_;
  MapT convMap_;
  MapT prodMap_;
};

template <typename Double>
Double computeKernel(const Double *const im1, const Double *const im2,
                     const size_t h, const size_t w, const size_t c,
                     const std::vector<size_t>& patchSizes,
                     const std::vector<size_t>& subs,
                     const std::vector<double>& sigmas,
                     const std::vector<int>& pools) {
  std::vector<LayerParams> layers;
  for (size_t i = 0; i < patchSizes.size(); ++i) {
    layers.push_back({patchSizes[i], subs[i], sigmas[i], /*zeroPad=*/true, static_cast<PoolType>(pools[i])});
  }

  CKNKernelMatrix<Double> kernel(layers, h, w, c);
  return kernel.computeKernel(im1, im2);
}
