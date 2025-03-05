#' Main function to run BARDDT model
#'
#' @param y Output variable
#' @param x Running variable
#' @param w Additional variables
#' @param c Cutoff value
#' @param delta Test set window
#' @param num_chains Number of parallel runs
#' @param num_gfr GFR samples
#' @param num_burnin Burnin
#' @param num_mcmc MCMC samples per chain
#' @param parallel bool: TRUE if running in parallel
#' @param ncores number of cores used for parallel computing
#'
#' @returns Posterior samples for the RDD CATE
#' @export
#'
#' @examples example
barddt <- function(y,x,w,c,delta=0.1,num_chains=20,num_gfr=2,num_burnin=0,num_mcmc=500,parallel=FALSE,ncores=10)
{
  x <- (x-c)/sd(x-c)
  test <- -delta < x & x < delta
  z <- as.numeric(x>0)
  ntest <- sum(test)
  ## We will sample multiple chains sequentially
  bart_models <- list()
  ## Define basis functions for training and testing
  B <- cbind(z*x,(1-z)*x, z,rep(1,n))
  B1 <- cbind(rep(c,n), rep(0,n), rep(1,n), rep(1,n))
  B0 <- cbind(rep(0,n), rep(c,n), rep(0,n), rep(1,n))
  B1 <- B1[test,]
  B0 <- B0[test,]
  B_test <- rbind(B1,B0)
  xmat_test <- cbind(x=rep(0,n),w)[test,]
  xmat_test <- rbind(xmat_test,xmat_test)
  ### We combine the basis for Z=1 and Z=0 to feed it to the BART call and get the Y(z) predictions instantaneously
  ### Then we separate the posterior matrix between each Z and calculate the CATE prediction
  ## Sampling trees in parallel
  if (parallel)
  {
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
    `%loop_operator%` <- `%dopar%`
  } else
  {
    `%loop_operator%` <- `%do%`
  }

  bart_model_outputs <- foreach (i = 1:num_chains) %loop_operator% {
    random_seed <- i
    ## Lists to define BARDDT parameters
    barddt.global.parmlist <- list(standardize=T,sample_sigma_global=TRUE,sigma2_global_init=0.1)
    barddt.mean.parmlist <- list(num_trees=50, min_samples_leaf=20, alpha=0.95, beta=2,
                                 max_depth=20, sample_sigma2_leaf=FALSE, sigma2_leaf_init = diag(rep(0.1/50,4)))
    bart_model <- stochtree::bart(
      X_train = cbind(x,w), leaf_basis_train = B, y_train = y,
      X_test = xmat_test, leaf_basis_test = B_test,
      num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc,
      general_params = barddt.global.parmlist, mean_forest_params = barddt.mean.parmlist
    )
    bart_model <- bart_model$y_hat_test[1:ntest,]-bart_model$y_hat_test[(ntest+1):(2*ntest),]
  }

  if (parallel) {
    stopCluster(cl)
  }
  pred <- do.call("cbind",bart_model_outputs)
}
