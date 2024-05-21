#include "../inst/include/AnnoyVQRecall.hpp"


// [[Rcpp::export(".find_BMU")]]
Rcpp::List find_BMU(const arma::mat& X, const arma::mat& W, unsigned int nBMU = 2, unsigned int nAnnoyTrees = 50, std::string QuantType = "hard", bool parallel = true) {
  arma::umat BMU;
  arma::mat QE;
  arma::mat ACT;

  cpp_findBMU(BMU, QE, ACT, // outputs, byref
              X, W, nBMU,
              parallel, nAnnoyTrees, QuantType);

  Rcpp::List out;
  out["BMU"] = BMU;
  out["QE"] = QE;
  out["ACT"] = ACT;

  return out;
}


// [[Rcpp::export(".Recall_BMU")]]
Rcpp::List Recall_BMU(unsigned int nW, const arma::umat& BMU, const arma::mat& QE, const arma::mat& ACT, bool parallel = true) {

  std::vector<arma::uvec> RF;
  arma::uvec RFSize;
  arma::vec RFQE_mean;
  arma::vec RFQE_sd;
  arma::vec RFQE_max;
  double Entropy;
  arma::sp_mat CADJ;
  arma::sp_mat fCADJ;
  arma::sp_mat CONN;

  cpp_Recall(RF, RFSize,
             RFQE_mean, RFQE_sd, RFQE_max, Entropy, CADJ, fCADJ, CONN,
             nW, BMU, QE, ACT, parallel);

  Rcpp::List out;
  out["RF"] = RF;
  out["RFSize"]  = RFSize;
  out["RFQE_mean"] = RFQE_mean;
  out["RFQE_sd"] = RFQE_sd;
  out["RFQE_max"]  = RFQE_max;
  out["Entropy"] = Entropy;
  out["CADJ"] = CADJ;
  out["fCADJ"] = fCADJ;
  out["CONN"] = CONN;

  return out;
}


// [[Rcpp::export(".RecallLabels_BMU")]]
Rcpp::List RecallLabels_BMU(const arma::uvec& XL, const std::vector<arma::uvec>& RF, const arma::umat& BMU, const arma::mat& ACT, bool parallel = true) {
  arma::uvec WL;  // prototype Labels
  arma::mat WL_Dist; // Frequency table of labels in each RF
  arma::vec WL_Purity; // purity score of each RF

  cpp_RecallLabels(WL, WL_Dist, WL_Purity,
                   XL, RF, BMU, ACT, parallel);

  Rcpp::List out;

  out["WL"] = WL;  // prototype Labels
  out["WL_Dist"] = WL_Dist; // Frequency table of labels in each RF
  out["WL_Purity"] = WL_Purity; // purity score of each RF

  return out;

}

// Rcpp::List Recall(const arma::mat& X, const arma::mat& W, const arma::uvec& XL, unsigned int nBMU, unsigned int nAnnoyTrees = 50, std::string QuantType = "hard", bool parallel = true) {
//
//   arma::umat BMU;
//   arma::mat QE;
//   arma::mat ACT;
//
//   cpp_findBMU(BMU, QE, ACT, // outputs, byref
//               X, W, nBMU,
//               parallel, nAnnoyTrees, QuantType);
//
//   std::vector<arma::uvec> RF;
//   arma::uvec RFSize;
//   arma::vec RFQE_mean;
//   arma::vec RFQE_sd;
//   arma::vec RFQE_max;
//   double Entropy;
//   arma::sp_mat CADJ;
//   arma::sp_mat CONN;
//
//   cpp_Recall(RF, RFSize,
//              RFQE_mean, RFQE_sd, RFQE_max, Entropy, CADJ, CONN,
//              W.n_rows, BMU, QE, ACT,
//              parallel);
//
//
//   arma::uvec WL;
//   arma::mat WL_Dist;
//   arma::vec WL_Purity;
//
//   cpp_RecallLabels(WL, WL_Dist, WL_Purity,
//                    XL, W.n_rows, RF, BMU, ACT, parallel);
//
//
//   Rcpp::List out;
//   out["BMU"] = BMU;
//   out["QE"] = QE;
//   out["ACT"] = ACT;
//   out["RF"] = RF;
//   out["RFSize"]  = RFSize;
//   out["RFQE_mean"] = RFQE_mean;
//   out["RFQE_sd"] = RFQE_sd;
//   out["RFQE_max"]  = RFQE_max;
//   out["Entropy"] = Entropy;
//   out["CADJ"] = CADJ;
//   out["CONN"] = CONN;
//
//   out["WL"] = WL;  // prototype Labels
//   out["WL_Dist"] = WL_Dist; // Frequency table of labels in each RF
//   out["WL_Purity"] = WL_Purity; // purity score of each RF
//
//   return out;
// }


