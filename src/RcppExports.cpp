// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// find_BMU
Rcpp::List find_BMU(const arma::mat& X, const arma::mat& W, unsigned int nBMU, unsigned int nAnnoyTrees, std::string QuantType, bool parallel);
RcppExport SEXP _AnnoyVQRecall_find_BMU(SEXP XSEXP, SEXP WSEXP, SEXP nBMUSEXP, SEXP nAnnoyTreesSEXP, SEXP QuantTypeSEXP, SEXP parallelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nBMU(nBMUSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nAnnoyTrees(nAnnoyTreesSEXP);
    Rcpp::traits::input_parameter< std::string >::type QuantType(QuantTypeSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    rcpp_result_gen = Rcpp::wrap(find_BMU(X, W, nBMU, nAnnoyTrees, QuantType, parallel));
    return rcpp_result_gen;
END_RCPP
}
// Recall_BMU
Rcpp::List Recall_BMU(unsigned int nW, const arma::umat& BMU, const arma::mat& QE, const arma::mat& ACT, bool parallel);
RcppExport SEXP _AnnoyVQRecall_Recall_BMU(SEXP nWSEXP, SEXP BMUSEXP, SEXP QESEXP, SEXP ACTSEXP, SEXP parallelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nW(nWSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type BMU(BMUSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type QE(QESEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type ACT(ACTSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    rcpp_result_gen = Rcpp::wrap(Recall_BMU(nW, BMU, QE, ACT, parallel));
    return rcpp_result_gen;
END_RCPP
}
// RecallLabels_BMU
Rcpp::List RecallLabels_BMU(const arma::uvec& XL, const std::vector<arma::uvec>& RF, const arma::umat& BMU, const arma::mat& ACT, bool parallel);
RcppExport SEXP _AnnoyVQRecall_RecallLabels_BMU(SEXP XLSEXP, SEXP RFSEXP, SEXP BMUSEXP, SEXP ACTSEXP, SEXP parallelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type XL(XLSEXP);
    Rcpp::traits::input_parameter< const std::vector<arma::uvec>& >::type RF(RFSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type BMU(BMUSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type ACT(ACTSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    rcpp_result_gen = Rcpp::wrap(RecallLabels_BMU(XL, RF, BMU, ACT, parallel));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_AnnoyVQRecall_find_BMU", (DL_FUNC) &_AnnoyVQRecall_find_BMU, 6},
    {"_AnnoyVQRecall_Recall_BMU", (DL_FUNC) &_AnnoyVQRecall_Recall_BMU, 5},
    {"_AnnoyVQRecall_RecallLabels_BMU", (DL_FUNC) &_AnnoyVQRecall_RecallLabels_BMU, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_AnnoyVQRecall(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
