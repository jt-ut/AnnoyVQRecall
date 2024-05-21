#ifndef RcppArmadillo_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef RcppParallel_H
#include <RcppParallel.h>
// [[Rcpp::depends(RcppParallel)]]
#endif

// [[Rcpp::plugins(cpp11)]]

#include "string.h"

#include "RcppAnnoy.h"
typedef double ANNOYTYPE;
typedef AnnoyIndex <int, ANNOYTYPE, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy> MyAnnoyIndex;



#ifndef ANNOYVQRECALL_BMU_HPP
#define ANNOYVQRECALL_BMU_HPP

// Function to turn QEs into Activations
inline arma::mat cpp_QE2Activation(const arma::mat& QE) {
  // // Add small number, in case some QE = 0 (don't want to dividy by 0 later)
  // arma::mat Act = QE + std::numeric_limits<double>::min();
  // arma::vec sumAct = arma::sum(Act, 1);
  // Act.each_col() /= sumAct;
  // Act = 1.0 - Act;
  // return Act;


  // // ** Compute activations using the QE[0] / (QE[0] + QE[i]) method
  // // Add small number, in case some QE = 0 (don't want to divide by 0 later)
  // arma::mat Act = QE + std::numeric_limits<double>::min();
  // for(unsigned int j=1; j<QE.n_cols; ++j) {
  //   Act.col(j) = Act.col(0) / (Act.col(0) + Act.col(j));
  // }
  // Act.col(0).ones();
  // arma::vec sumAct = arma::sum(Act, 1);
  // Act.each_col() /= sumAct;
  // return Act;


  // ** Compute activations using negative exponential
  arma::mat Act = arma::exp(-QE);
  arma::vec sumAct = arma::sum(Act, 1);
  Act.each_col() /= sumAct;
  return Act;

}


// Parallel worker to find BMU of data
struct AnnoyBMU_prlwkr : public RcppParallel::Worker {

  // Inputs
  const arma::mat& X;
  const arma::mat& W;
  unsigned int nBMU;
  unsigned int nAnnoyTrees;
  std::string QuantType; // either "hard" or "soft", "soft" splits


  // Intermediaries
  unsigned int ndims;
  unsigned int nX;
  unsigned int nW;
  MyAnnoyIndex AnnoyObj;

  // Outputs
  arma::umat BMU;
  arma::mat QE;
  arma::mat ACT; // activations


  AnnoyBMU_prlwkr(const arma::mat& X, const arma::mat& W, unsigned int nBMU, unsigned int nAnnoyTrees, std::string QuantType) :
    X(X), W(W), nBMU(nBMU), nAnnoyTrees(nAnnoyTrees), QuantType(QuantType),
    ndims(X.n_cols), nX(X.n_rows), nW(W.n_rows), AnnoyObj(ndims)
  {
    if(W.n_cols != ndims) Rcpp::stop("ncol(X) != ncol(W)");
    if(nBMU > nW) Rcpp::stop("nBMU > nrow(W)");

    BMU = arma::zeros<arma::umat>(nX,nBMU);
    QE = arma::zeros<arma::mat>(nX,nBMU);
    ACT = arma::zeros<arma::mat>(nX,nBMU);
    //std::transform(QuantType.begin(), QuantType.end(), QuantType.begin(), std::tolower);
    std::transform(QuantType.begin(), QuantType.end(), QuantType.begin(),
                   [](unsigned char c){ return std::tolower(c); } );

    this->build_AnnoyObj();
  };


  void build_AnnoyObj() {
    arma::rowvec tmp;
    for(unsigned int i=0; i<nW; ++i) {
      tmp = W.row(i);
      AnnoyObj.add_item(i, tmp.memptr());
    }
    AnnoyObj.build(nAnnoyTrees);
  }


  void find_BMU_of_Xi(unsigned int i) {

    std::vector<int> nhb_index;
    std::vector<ANNOYTYPE> nhb_dist;

    arma::rowvec tmp = X.row(i);
    AnnoyObj.get_nns_by_vector(tmp.memptr(), nBMU, -1, &nhb_index, &nhb_dist);


    for(unsigned int k=0; k<nBMU; ++k) {
      BMU(i,k) = (unsigned int)nhb_index[k];
      QE(i,k) = nhb_dist[k];
    }

    if(QuantType=="hard") {
      ACT.row(i).zeros();
      ACT(i,0) = 1.0;
    } else {
      ACT.row(i) = arma::exp(-QE.row(i));
      ACT.row(i) /= arma::accu(ACT.row(i));
    }


    return;
  }

  // process a block of x
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++) {
      find_BMU_of_Xi(i);
    }
  }


  void calc_parallel() {
    RcppParallel::parallelFor(0, nX, *this);
  }

  void calc_serial() {
    for(unsigned int i=0; i<X.n_rows; ++i) {
      find_BMU_of_Xi(i);
    }
  }

};

// Function to find BMU of data
inline void cpp_findBMU(arma::umat& BMU, arma::mat& QE, arma::mat& ACT, // outputs, byref
                         const arma::mat& X, const arma::mat& W, unsigned int nBMU = 2,
                         bool parallel = true, unsigned int nAnnoyTrees = 50, std::string QuantType = "hard") {

  AnnoyBMU_prlwkr wkr(X, W, nBMU, nAnnoyTrees, QuantType);

  if(parallel) wkr.calc_parallel(); else wkr.calc_serial();

  BMU = wkr.BMU;
  QE = wkr.QE;
  ACT = wkr.ACT;

  return;
}



// Parallel worker to perform Recall, assuming BMU & QE have already been calculated.
// This populates all RF-level quantities, other than label-based info.
struct VQRecall_worker : public RcppParallel::Worker {

  // Inputs
  unsigned int nW; // # of prototypes
  const arma::umat& BMU; // 0-based indices of BMUs of data points
  const arma::mat& QE; // quantization error
  const arma::mat& ACT;


  // output containers
  std::vector<arma::uvec> RF;
  arma::uvec RFSize;
  arma::vec RFQE_mean;
  arma::vec RFQE_sd;
  arma::vec RFQE_max;
  double Entropy;
  arma::sp_mat CADJ;
  arma::sp_mat fCADJ;
  arma::sp_mat CONN;


  // Constructor
  VQRecall_worker(unsigned int nW, const arma::umat& BMU, const arma::mat& QE, const arma::mat& ACT)
    : nW(nW), BMU(BMU), QE(QE), ACT(ACT)
  {

    // Make sure BMU & QE have the same size, and that BMU has at least 2 columns to compute CADJ
    if(BMU.n_rows != QE.n_rows) Rcpp::stop("size(BMU) != size(QE)");
    if(BMU.n_cols != QE.n_cols) Rcpp::stop("size(BMU) != size(QE)");
    //if(BMU.n_cols < 2) Rcpp::stop("ncols(BMU) must be >= 2");

    // Compute Activations
    //Activations = cpp_QE2Activation(QE);

    // Initialize Containers
    RF.resize(nW);
    RFSize.set_size(nW); RFSize.zeros();
    Entropy = 0;
    RFQE_mean.set_size(nW); RFQE_mean.zeros();
    RFQE_sd.set_size(nW); RFQE_sd.zeros();
    RFQE_max.set_size(nW); RFQE_max.zeros();

    CADJ.set_size(nW, nW); CADJ.zeros();
    fCADJ.set_size(nW, nW); fCADJ.zeros();

  }


  // Fill up the containers for a single prototype
  void process_single_prototype(unsigned int i) {

    // Find RF members. If RF is empty, return.
    RF[i] = arma::find(BMU.col(0) == i);
    if(RF[i].size() == 0) return;

    // Sort the RF, just for cleanliness
    RF[i] = arma::sort(RF[i]);

    // Set the size
    RFSize(i) = RF[i].size();

    // Compute mean quantization error within RF
    RFQE_mean(i) = arma::mean(QE.elem(RF[i]));

    // Standard deviation of quantization error within RF
    // The last argument to stddev is 'norm_type' = denominator used to compute sd.
    // If we have > 1 points, use unbiased estimator (denominator = N-1), otherwise use N
    if(RFSize(i) > 1) {
      RFQE_sd(i) = arma::stddev( QE.elem(RF[i]), 0 );
    } else {
      RFQE_sd(i) = 0; //arma::stddev( QE.elem(RF[i]), 1 );
    }

    // Radius of RF = max QE
    RFQE_max(i) = arma::max( QE.elem(RF[i]) );

    // Fill up CADJ, if there are at least 2BMUs given
    if(BMU.n_cols >= 2) {
      for(unsigned int j=0; j<RFSize(i); ++j) {

        // arma::vec fuzzy_act(BMU.n_cols - 1); // no CADJ activation for winner prototype
        // for(unsigned int k=0; k<BMU.n_cols-1; ++k) {
        //   fuzzy_act(k) = Activations(RF[i][j], k+1);
        // }
        // fuzzy_act /= arma::accu(fuzzy_act); // re-normalize
        //
        // for(unsigned int k=0; k<BMU.n_cols-1; ++k) {
        //   CADJ(i, BMU( RF[i](j), k+1)) += fuzzy_act[k];
        // }


        for(unsigned int k=0; k<BMU.n_cols-1; ++k) {
          CADJ(i, BMU( RF[i](j), k+1)) += 1.0;
          fCADJ(i, BMU( RF[i](j), k)) += ACT(i, k);
        }
        fCADJ(i, BMU( RF[i](j), BMU.n_cols-1)) += ACT(i, BMU.n_cols-1);

      }

      return;

    }
  }

  // Parallel operator - find BMU of each row of X in parallel
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++) {
      process_single_prototype(i);
    }
  }

  // Parallel call method
  void calc_parallel() {
    RcppParallel::parallelFor(0, nW, *this);
    this->calc_global_measures();
    CONN = CADJ + CADJ.t();
  }

  // Non-parallel call method
  void calc_serial() {
    // Find BMU of each row of X
    for(unsigned int i=0; i<nW; ++i) {
      process_single_prototype(i);
    }
    this->calc_global_measures();
    CONN = CADJ + CADJ.t();
  }


  void calc_global_measures() {

    // Strip out the active RFs (those with nonzero size)
    arma::uvec activeRF = arma::find(RFSize);
    //unsigned int nactiveRF = activeRF.n_elem;

    // *** Normalized entropy of the quantization
    // Can only do this for active RFs to avoid log(0) in the calculation
    arma::vec pRF = arma::conv_to<arma::vec>::from(RFSize.elem(activeRF)) / double(arma::accu(RFSize));
    Entropy = -arma::accu(pRF % arma::log(pRF)) / std::log(nW);

    // If no labeled data, return
    return;
  }

  void clear() {
    // Clear out the recall containers
    for(unsigned int j=0; j<nW; ++j) {
      RF[j].clear();
    }

    RFSize.zeros();
    RFQE_mean.zeros();
    RFQE_sd.zeros();
    RFQE_max.zeros();
    Entropy = 0.0;
    CADJ.zeros();
    fCADJ.zeros();
    CONN.zeros();
  }
};


inline void cpp_Recall(std::vector<arma::uvec>& RF, arma::uvec& RFSize,
                       arma::vec& RFQE_mean, arma::vec& RFQE_sd, arma::vec& RFQE_max,
                       double& Entropy,
                       arma::sp_mat& CADJ, arma::sp_mat& fCADJ, arma::sp_mat& CONN,
                       unsigned int nW, const arma::umat& BMU, const arma::mat& QE, const arma::mat& ACT,
                       bool parallel = true) {

  VQRecall_worker recworker(nW, BMU, QE, ACT);

  if(parallel) {
    recworker.calc_parallel();
  } else {
    recworker.calc_serial();
  }

  RF = recworker.RF;
  RFSize = recworker.RFSize;
  RFQE_mean = recworker.RFQE_mean;
  RFQE_sd = recworker.RFQE_sd;
  RFQE_max = recworker.RFQE_max;
  Entropy = recworker.Entropy;
  CADJ = recworker.CADJ;
  fCADJ = recworker.fCADJ;
  CONN = recworker.CONN;

  return;

}

// Parallel worker to perform Recall of data Labels, assuming BMU & QE have already been calculated.

struct VQRecallLabels_worker : public RcppParallel::Worker {

  // Inputs
  const arma::uvec& XL; // X labels, encoded as consecutive integers starting from 1 (e.g., as produced by factor)
  unsigned int nW; // # of prototypes
  const std::vector<arma::uvec>& RF;
  const arma::umat& BMU; // 0-based indices of BMUs of data points
  const arma::mat& ACT; // activations for BMUs

  arma::uvec XL_unq; // Unique set of labels in X
  std::map<unsigned int, unsigned int> labelmap; // <label, factor level>

  // output containers
  arma::uvec WL;  // prototype Labels
  arma::mat WL_Dist; // Frequency table of labels in each RF
  arma::vec WL_Purity; // purity score of each RF

  // Constructor
  VQRecallLabels_worker(const arma::uvec& XL, unsigned int nW, const std::vector<arma::uvec>& RF, const arma::umat& BMU, const arma::mat& ACT)
    : XL(XL), nW(nW), RF(RF), BMU(BMU), ACT(ACT)
  {

    // Make sure BMU & ACT have the same size,
    if(BMU.n_rows != ACT.n_rows) Rcpp::stop("size(BMU) != size(ACT)");
    if(BMU.n_cols != ACT.n_cols) Rcpp::stop("size(BMU) != size(ACT)");


    // Determine unique set of input labels,
    // create a <unique label, factor index> map.
    XL_unq = arma::sort(arma::unique(XL));
    for(unsigned int k=0; k<XL_unq.size(); ++k) {
      labelmap[XL_unq[k]] = k;
    }

    // Initialize Containers
    WL.resize(nW); WL.zeros();
    WL_Dist.set_size(nW, XL_unq.size()); WL_Dist.zeros();
    WL_Purity.resize(nW); WL_Purity.zeros();
  }


  VQRecallLabels_worker(const VQRecallLabels_worker& me, RcppParallel::Split) :
    XL(me.XL), nW(me.nW), RF(me.RF), BMU(me.BMU), ACT(me.ACT),
    XL_unq(me.XL_unq),
    labelmap(me.labelmap),
    WL(me.WL),
    WL_Dist(arma::zeros<arma::mat>(nW,labelmap.size())),
    WL_Purity(me.WL_Purity)
  {};




  void contribution_from_label_i(unsigned int i) {

    // for(unsigned int k=0; k<BMU.n_cols; ++k) {
    //   WL_Dist(BMU(i,k), XL[i]-1) += Activations(i, k);
    // }

    unsigned int lblidx = labelmap[XL[i]];
    for(unsigned int k=0; k<BMU.n_cols; ++k) {
      WL_Dist(BMU(i,k), lblidx) += ACT(i, k);
    }

  }

  // join my values with that of another thread
  void join(const VQRecallLabels_worker& rhs) {
    WL_Dist += rhs.WL_Dist;
  }

  // Parallel operator - find BMU of each row of X in parallel
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++) {
      contribution_from_label_i(i);
    }
  }

  // Use WL_Dist to determine winner
  void find_winning_label() {
    for(unsigned int i=0; i<nW; ++i) {
      double rowsum = arma::accu(WL_Dist.row(i));
      if(!(rowsum > std::numeric_limits<double>::min())) continue;

      // Normalize this WLDIst
      WL_Dist.row(i) /= rowsum;
      // Find winner
      arma::uword winner_idx = WL_Dist.row(i).index_max();
      WL[i] = XL_unq[winner_idx];

      // Hellinger distance between norm_dist and a "perfect" distribution, where norm_dist[winner.idx] = 1 and norm_dist[all others] = 0
      double hell_dist = std::sqrt(1.0 - std::sqrt(WL_Dist(i, winner_idx)));
      WL_Purity[i] = 1.0 - hell_dist; // 1 - hell_dist is measure of similarity

    }
  }

  // Parallel call method
  void calc_parallel() {
    RcppParallel::parallelReduce(0, XL.size(), *this);
    this->find_winning_label();
  }

  // Non-parallel call method
  void calc_serial() {
    // Find BMU of each row of X
    for(unsigned int i=0; i<XL.size(); ++i) {
      contribution_from_label_i(i);
    }
    this->find_winning_label();
  }

};

inline void cpp_RecallLabels(arma::uvec& WL, arma::mat& WL_Dist, arma::vec& WL_Purity,
                             const arma::uvec& XL, const std::vector<arma::uvec>& RF, const arma::umat& BMU, const arma::mat& ACT,
                             bool parallel = true) {

  unsigned int nW = RF.size();
  VQRecallLabels_worker lblworker(XL, nW, RF, BMU, ACT);

  if(parallel) {
    lblworker.calc_parallel();
  } else {
    lblworker.calc_serial();
  }

  WL = lblworker.WL;  // prototype Labels
  WL_Dist = lblworker.WL_Dist; // Frequency table of labels in each RF
  WL_Purity = lblworker.WL_Purity; // purity score of each RF

  return;
}



#endif
