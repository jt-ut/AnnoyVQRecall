findBMU = function(X, W, nBMU = 2, nAnnoyTrees = 50, QuantType = "hard", parallel = TRUE) {
  ## Input checks:
  # nBMU & nAnnoyTrees must be > 0
  stopifnot(nBMU > 0)
  stopifnot(nAnnoyTrees > 0)
  # ncol(X) should = ncol(W)
  stopifnot(ncol(X) == ncol(W))
  # QuantType must be "hard" or "soft"
  QuantType = tolower(QuantType)
  stopifnot(QuantType %in% c('hard','soft'))

  out = AnnoyVQRecall:::.find_BMU(X=X, W=W, nBMU=nBMU, nAnnoyTrees = nAnnoyTrees, QuantType = QuantType)

  ## Re-adjust the BMU indices to be 1-indexed (they are 0-indexed coming from C++)
  out$BMU = out$BMU + 1

  return(out)
}

VQRecall = function(X = NULL, W = NULL, nBMU = 2, XL = NULL,
                    nW = NULL, BMU = NULL, QE = NULL, ACT = NULL,
                    nAnnoyTrees = 50, QuantType = "hard", parallel = TRUE) {

  ## Input checks:
  # nBMU & nAnnoyTrees must be > 0
  stopifnot(nBMU > 0)
  stopifnot(nAnnoyTrees > 0)
  # QuantType must be "hard" or "soft"
  QuantType = tolower(QuantType)
  stopifnot(QuantType %in% c('hard','soft'))


  ## Option 1:
  # Performs full recall (find BMU, fill up RF containers, add label info if XL is given)
  # Mandatory Inputs: X, W, nBMU
  # Optional Inputs: XL
  inputX = !is.null(X)
  inputW = !is.null(W)
  use_option1 = inputX && inputW

  if(use_option1) {
    stopifnot(ncol(X) == ncol(W))
    bmu = AnnoyVQRecall:::.find_BMU(X=X, W=W, nBMU=nBMU, nAnnoyTrees = nAnnoyTrees, QuantType = QuantType) # returns BMU info 0-indexed
    rec = AnnoyVQRecall:::.Recall_BMU(nW = nrow(W), BMU = bmu$BMU, QE = bmu$QE, ACT = bmu$ACT, parallel = parallel) # returns RF info 0-indexed
    out = c(bmu, rec)
    rm(bmu, rec)
  }


  ## Option 2:
  # Performs recall given pre-computed BMU + QE + ACT
  inputnW = !is.null(nW)
  inputBMU = !is.null(BMU)
  inputQE = !is.null(QE)
  inputACT = !is.null(ACT)
  use_option2 = inputnW && inputBMU && inputQE && inputACT

  if(use_option2) {
    # BMU indices coming from R are 1-indexed, temporarily make 0-indexed
    out = list("BMU" = BMU-1, "QE" = QE, "ACT" = ACT)
    rec = AnnoyVQRecall:::.Recall_BMU(nW = nW, BMU = out$BMU, QE = out$QE, ACT = out$ACT, parallel = parallel)
    out = c(out, rec)
    rm(rec)
  }

  ## Make sure that either option1 or option2 were selected
  if(!use_option1 && !use_option2) {
    stop("Recall requires inputting either {X, W} or {nW, BMU, QE, ACT}")
  }

  ## Label recall, if labels were given
  if(!is.null(XL)) {
    # Ensure they're the right size
    if(length(XL) != nrow(out$BMU)) stop("length(XL) incorrect")
    # Factorize them
    XL = factor(XL)
    # Recall, combine
    reclbl = AnnoyVQRecall:::.RecallLabels_BMU(XL = XL, RF = out$RF, BMU = out$BMU, ACT = out$ACT, parallel = parallel)
    out = c(out, reclbl)
    rm(reclbl)
  }

  ## Re-adjust the BMU & RF indices to be 1-indexed (they are 0-indexed coming from C++)
  out$BMU = out$BMU + 1
  out$RF = lapply(out$RF, function(z) z+1)

  return(out)
}
