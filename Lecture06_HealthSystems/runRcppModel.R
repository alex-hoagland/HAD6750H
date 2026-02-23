# still need to incorporate new correlations from survey data and check Trello
# each iteration takes 45-60 seconds depending on available memory and cores6
# standard errors take at least a week to calculate
# initial GMM (one-shot) takes <12 hours 
# two-step GMM takes XXX
# simulated annealing takes XXX

rm(list = ls())
setwd("C:/Users/alexh/Dropbox/LungCancerScreening/LungCancerScreeningCode/") # Alex directory
# setwd("~/Dropbox/Research/Smoking/Screening/Code")  # Michael directory

# load("C:/Users/alexh/Dropbox/LungCancerScreening/LungCancerScreeningCode/RcppModel_GMM_Estimation_20260130.RData") # last results
################################################################################


# Libraries
################################################################################
library(tidyverse)
library(beepr) # lets me know when the code is done running
library(Rcpp)
library(gmm)
library(mvtnorm)      # rmvnorm
library(truncnorm)    # qtruncnorm

set.seed(12032025)

sourceCpp("Model_Rcpp.cpp")# , verbose=TRUE) # main model code
sourceCpp("calculateMoments_individual.cpp")# , verbose=TRUE) # compute the moments
################################################################################

################################################################################
# Simulation sample
################################################################################

################################################################################
# start with N individuals at age 40 with smoking histories
N <- 100000 # forward simulation is quite quick after solving the model 

# --- target correlations among (h_i latent, theta latent, phi latent)
R <- matrix(c(
  1.0000, 0.0421, 0.1595,
  0.0421, 1.0000, 0.1075,
  0.1595, 0.1075, 1.0000
), nrow = 3, byrow = TRUE)
if (any(eigen(R, symmetric = TRUE)$values <= 0)) stop("Correlation matrix not positive definite.")

# latent correlated normals
Z <- rmvnorm(n = N, mean = c(0, 0, 0), sigma = R)

# convert to uniforms via Gaussian copula
U <- pnorm(Z)

# --- marginals you want
# theta prevalence: based on alpha regression
p_theta2 <- 0.00125

# phi prevalence: based on phi regression for smokers at age 40 (note: this is for COPD + CVD only)
p_phi2 <- 0.0544

# h_i marginal: truncated normal 1..(40-12)=28 with mean/sd as given, then floor + clamp
h_min <- 1
h_max <- 40 - 12   # 28
h_mean <- 10.95
h_sd   <- 11.30

sampledata <- data.frame(
  id = 1:N,
  age = rep(40, N),
  # Use copula U[,1] -> truncated normal quantile -> discretize/clamp
  h_i = {
    x <- qtruncnorm(U[,1], a = h_min, b = h_max, mean = h_mean, sd = h_sd)
    x <- floor(x)
    pmin(pmax(h_min, x), h_max)
  },
  # Use copula U[,2] to set theta
  theta = ifelse(U[,2] <= (1 - p_theta2), 1L, 2L),
  # Use copula U[,3] to set phi
  phi = ifelse(U[,3] <= (1 - p_phi2), 1L, 2L), 
  stigma = sample(c(1,2), N, replace=TRUE, prob=c(1-0.475, 0.475)), 
  optimism = sample(c(1,2), N, replace=TRUE, prob=c(0.44, 0.56))
)

# check achieved correlations (note: these are correlations on realized variables,
# so they won't match exactly, especially with rare binary theta)
# cor(sampledata[, c("h_i", "theta", "phi")])

# NOTES on sample data:
# sampledata <- data.frame(
#   id = 1:N,
#   age = rep(40, N),
#   h_i = pmin(pmax(1,floor(rnorm(n=N,mean=10.95,sd=11.30))), 40-12), # smoking history between 1 and 40-12 pack-years (mean/SD from BRFSS 2024 for 40-yo smokers)
#   theta = sample(c(1,2), N, replace=TRUE, prob=c(1-0.0086,0.0086)), # majority in no cancer, very few in early cancer groups; 0 chance of late-stage cancer to start (mean/SD from BRFSS 2024 for 40-yo smokers)
#   phi = sample(c(1,2), N, replace=TRUE, prob=c(1-0.5085,0.5085)), # 30% chance of having a chronic condition at age 40
#   # chronic conditions include diabetes, heart attack, angina/CHD, stroke, asthma, COPD, cancer, kidney disease, depression
# take out diabetes, asthma, depression (double check that this << 50%). Make sure that mortality coefficient is updated (bigger) accordingly. 
#   stigma = sample(c(1,2), N, replace=TRUE, prob=c(1-0.475, 0.475)), # 47.5% chance of high stigma based on Prolific survey data
#   optimism = sample(c(1,2), N, replace=TRUE, prob=c(0.44, 0.56)) # 44% are not optimistic, 56% are optimistic 
#     # Average beliefs here (true = 12/100) =32/100 for the low-quality folks and 5.8 for the high-quality beliefs  
# )

# target correlations for initial draws (TODO: how to pull in stigma, optimism, and shock correlations)
# corr(h_i, theta) = 0.0421
# corr(h_i, phi) = 0.1595
# corr(theta, phi) = 0.1075

# draw shocks for beliefs just once (TODO: how to pull in stigma, optimism, and shock correlations)
sampledata <- sampledata %>% mutate(shock = rnorm(N,0,1), 
                                    type = ifelse(optimism == 1 & stigma == 1, 1,  
                                                  ifelse(optimism == 1 & stigma == 2, 2, 
                                                         ifelse(optimism == 2 & stigma == 1, 3, 4))))
# Types: 1 = low optimism, low stigma (baseline + optimism effect)
#        2 = low optimism, high stigma (baseline + stigma effect)
#        3 = high optimism, low stigma (baseline)
#        4 = high optimism, high stigma (baseline + optimism effect + stigma effect + interaction term)
################################################################################


################################################################################
# Parameters
################################################################################

simulate_moment <- function(structparams, data=sampledata) {
  
  # for debugging purposes only 
  # structparams <- c(4.88e05,  2.04e+04, -9.67e+05, -3.65e+05,  9.87e+04,  3.34e+04,  34.52,  55.72, 9.34e04, 4.50e+03,  1.50e+03,  1.11e+04,  1500,  3.85e+03, 3.85e+03,  5.26e-05,  9.34e-02)
  
  T <- 100         # Maximum age (we start at age 40 and run until age 100)
  tmin <- 40       # Starting Age
  beta <- 0.95     # Discount Factor
  
  # CRRA utility function parameters
  gamma <- c(structparams[1], structparams[2]) # Intercept to ensure positive utility and dynamic health incentives
  gamma0 <- c(2, 2) #         # CRRA parameter for theta<3 and theta=3
  
  gamma1 <- c(structparams[3], structparams[4]) # Direct MU of smoking  for theta<3 and theta=3
  gamma2 <- c(structparams[5], structparams[6]) # Reinforcement of smoking history (linear) for theta<3 and theta=3
  gamma3 <- c(structparams[7], structparams[8]) # Reinforcement of smoking history (quadratic) for theta<3 and theta=3
  
  gamma4 <- c(structparams[9]+structparams[11], structparams[9]+structparams[10]+structparams[11]+structparams[12], structparams[9], structparams[9]+structparams[10]) # Marginal Disutility of screening
  # Types: 1 = low optimism, low stigma (baseline + optimism effect)
  #        2 = low optimism, high stigma (baseline + optimism effect + stigma effect + interaction term)
  #        3 = high optimism, low stigma (baseline)
  #        4 = high optimism, high stigma (baseline + stigma effect)
  
  gamma5 <- structparams[13] # captures the marginal disutility from spending on screening
  # NOTES: - screening doesn't occur if theta = 3, so we don't need this to vary across theta
  # Note: this expected spending is (1-elig)*px + pr(screen positive)*px
  # Identification: based on the *price sensitivity* moments in the data 
  
  gamma6 <- structparams[14] # disutility of high cancer risk (incentive to quit)
  # NOTES: this applies with and without smoking -- disutility of increased risk over time (+ incentive to screen)
  #        - this should vary across theta as a disutility of cancer (where bt=1 --> worse than being at risk)
  # these are identified through indirect inference/regression coefficients between pack-history and smoking prevalence (BRFSS)
  
  gamma7 <- structparams[15] # disutility of symptoms (incentive to screen), where symptoms is a binary variable
  # NOTES: likelihood of symptoms varies widely with stage (much more common in late stage) so no need to let this vary by theta=3?
  # Identification is based on external moments suggesting the impact of symptoms on screening likelihood 
  
  # Initial guesses for the GMM model: mean and sd of beliefs *about cancer* at age 40
  # Starting with a lognormal distribution centered at 0.05 with sd 0.02
  sampledata <- sampledata %>%
    mutate(
      b_i = exp((log(structparams[16])-0.5*log(1+(structparams[17]^2/structparams[16]^2))) + log(1+(structparams[17]^2/structparams[16]^2))*shock)
    ) %>%
    mutate(b_i = pmin(pmax(b_i, 0), 1)) # cap beliefs at 0 or 1
  
  # Note: beliefs are fixed at 1 when theta = 3 (since beliefs are about early-stage cancer)
  # For theta = 2 (early stage cancer), theta is still unknown. 
  # If there is a positive test, we assume there is confirmatory testing so uncertainty is 
  # temporarily resolved, but evolves according to rational expectations afterwards (becomes uncertain again)
  
  # Prices and Income
  ps <- 9*365                # Price of cigarettes ($9 pack per day for 1 year, using 2022-2025 data and ChatGPT)
  px <-  300                 # Price of screening OOP *if* one is ineligible for USPSTF free screening (also this is the price of confirmatory testing 100% of the time)
  py <- 5000                 # Price of treatment OOP for one year (estimated using https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2836518?utm_source=chatgpt.com).
  w <- 63500                # Annual Income given our survey
  
  # Screening Quality Parameters
  # 44% have beliefs that are higher than 12/100 and 56% have beliefs that are lower than 12
  # 32/100 for the low-quality folks and 5.8 for the high-quality beliefs  
  rho <- 0.703                # Probability of true positive (from Truveta)
  rhohat <- 0.63501       # Subjective belief of true positive (from survey)
  lambda <-   0.121             # Probability of false positive (from Truveta) -- note: not specificity!
  lambdahat <- 0.17693        # Subjective belief of false positive (from survey) -- note: not specificity!
  
  # Health Transition Parameters 
  alpha <- c(-8.37, 0.5269, 1.31, 0.8189)         # Cancer onset (theta=1-->theta=2): intercept, smoking history (10 pack years), smoking dummy, age (decades over 40)
  zeta <- c(-4.5951, 2.2976, 1.2611, -1.9080) # Progression from 2 to 3 (intercept, years since early stage, treatment, treatment * years since early stage)
  psi  <- c(-4.079, 1.223, 0.7831, -0.3094)       # CVD and COPD Other health progression 1 to 2 (intercept, smoking, age in decades, smoking * age )
  omega  <- c(-5.397, 1.37, 3.569, 0.58, 0.7653)  # Mortality: intercept, early-stage cancer, late stage cancer, other chronic condition, age
  # kappa <- c(-4.30, 0.35, 0.80, 0.45) # symptom detection of late-stage cancer only (right now, targeted to ~25% of those with theta=3 identify this way in a given year)
  
  b_grid <- seq(0, 1, length.out = 101) # initial beliefs about early stage lung cancer
  h_grid <- 1:101 # smoking history -- run this up to 100 to accommodate very elderly heavy smokers
  n_b <- length(b_grid)
  n_h <- length(h_grid)
  n_theta <- 3
  ################################################################################
  
  
  # run the model
  # start <- Sys.time()
  intermediate <- solve_model(T, tmin, b_grid, h_grid,beta,
                        gamma, gamma0,gamma1,gamma2,gamma3,gamma4,gamma5,gamma6,gamma7,
                        ps, px, py, w, rhohat, lambdahat, rho, lambda, alpha, zeta, psi, omega)
  # print( Sys.time() - start ) # takes ~60seconds to run on my machine with parallelization
  
  # stores value function and policy function across the grids (including a separate one for early stage cancer that takes into account years since dx)
  # note that V is stored as an array with dimensions c(n_b, n_h, 3 (theta), 2 (phi), T+1, d (diagnosis))
  # note that policy is stored as an array with dimensions c(n_b, n_h, 3 (theta), 2 (phi), 2 (one for smoking and one for screening), T+1, 2 (diagnosis states))
  ################################################################################
  
  
  #### Now need to simulate outcomes for a cohort of N individuals starting at age 40 
  sim_df <- simulate_cohort_cpp(
    N         = N,
    start_age = 40,
    max_age   = 99,
    b_grid    = seq(0, 1, 0.01),
    h_grid    = 1:101,
    policy    = intermediate$policy,
    policy2    = intermediate$policy2,
    alpha     = alpha,
    zeta      = zeta,
    psi       = psi,
    omega     = omega,
    rhohat    = rhohat,
    lambdahat = lambdahat,
    rho       = rho, 
    lambda    = lambda,
    b0        = sampledata$b_i,
    h0        = sampledata$h_i,
    theta0    = sampledata$theta,
    phi0      = sampledata$phi, 
    type      = sampledata$type
  )
  ################################################################################
  
  # use the calculateMoments.R function to calculate and return the vector of moment differences
  sim_df <- sim_df %>% filter(age <= 80 & age >= 40) #  just calculate moments on ages 40-80
  moment_diffs <- moment_diffs_cpp(
    age   = sim_df$age,
    s     = sim_df$s,
    x     = sim_df$x,
    h     = sim_df$h,
    dead  = sim_df$dead,
    theta = sim_df$theta, 
    type  = sim_df$type
  ) 
  
  return(moment_diffs)
}

#### embed into a GMM structure where we select gamma parameters and belief parameters to minimize moment_diffs
moment_function <- function(structparams, sampledata) {
  # Wrap the call to compute_f in tryCatch
  result <- tryCatch({
    # Attempt to compute the moments normally
    simulate_moment(structparams, sampledata)
  }, error = function(e) {
    # In case of an error, return a penalty value for the N x 12 moment matrix 
    penalty_value <- 1e4
    matrix(penalty_value, nrow = N*32, ncol = 29)
  })
  
  return(result)
}
###########################################################


##### 4. Estimate using GMM #####
# if you want to, start by fixing some of the parameters to reduce dimension at first
# also used to bound parameter space (to positive values for some parameters)
# modelnum indicates which of the models you are estimating (1 = base model; 2 = add gamma 4-6; 3 = let beliefs fluctuate as well)
reduce_dimension <- function(theta_free, sampledata, modelnum=3) {
  set.seed(12032025)
  theta_full <- c(
    0+(1000000-0)*(1/(1+exp(-theta_free[1]))),  # gamma intercept 1 -- should be positive
    0+(1000000-0)*(1/(1+exp(-theta_free[2]))),  # gamma intercept 2 -- should be positive
    -1000000+(100--1000000)*(1/(1+exp(-theta_free[3]))),  # gamma1 for theta<3 -- bounded above by 0 
    -1000000+(100--1000000)*(1/(1+exp(-theta_free[4]))),  # gamma1 for theta=3 -- bounded above by 0
    0+(1000000-0)*(1/(1+exp(-theta_free[5]))),  # gamma2 for theta<3 -- should be positive 
    0+(1000000-0)*(1/(1+exp(-theta_free[6]))),  # gamma2 for theta=3 -- should be positive 
    0+(100-0)*(1/(1+exp(-theta_free[7]))),  # gamma3 for theta<3 -- Fix to be small positive 
    0+(100-0)*(1/(1+exp(-theta_free[8]))),  # gamma3 for theta=3 -- Fix to be small positive 
    -1000+(100000--1000)*(1/(1+exp(-theta_free[9]))),     # base disutility (high optimism, low stigma) 
    -100+(100000--100)*(1/(1+exp(-theta_free[10]))),            # extra hit from high stigma  
    -100+(100000--100)*(1/(1+exp(-theta_free[11]))),            # extra hit from low optimism 
    0+(100000-0)*(1/(1+exp(-theta_free[12]))),            # interaction effect from both hitting  
    -10000+(100000-(-10000))*(1/(1+exp(-theta_free[13]))),            # gamma5 when theta < 3  -- let this range between -10 and 10
    0+(10000-0)*(1/(1+exp(-theta_free[14]))), # gamma6  -- fix to be between 0 and 10000
    0+(10000-0)*(1/(1+exp(-theta_free[15]))), # gamma7 -- fix to be between 0 and 10000
    0+(0.1-0)*(1/(1+exp(-theta_free[16]))),  # belief mean (log scale) -- # Fix belief parameters
    0+(0.1-0)*(1/(1+exp(-theta_free[17])))  # belief sd (log scale)
  ) 
  
  G <- moment_function(theta_full, sampledata) # scale by SD  
  G <- sweep(G, 2, s0, "/")
  
  return(G) # scaled moment_function
}

# Types: 1 = low optimism, low stigma
#        2 = low optimism, high stigma
#        3 = high optimism, low stigma
#        4 = high optimism, high stigma

scale_ratio <- c(1, 0.7, 7, 1.25, 8.5, 6.5, 1, 1.5, 0.75, 0.05, 0.1, 0.35, 6.25,
                 2, 2.75, 3, 1.5)

# scale the columns of G 

# note: this uses the back-transformed values to set the base
G0 <- moment_function(c(2.56e5,  3.32e5, -5.87e2, -2.31e5,  2.244e2,  9.986e5,  27.9,  82.49, 3.26e4,
                        4.86e+04,  4.4757e4, 4.14e4, 9.978e4, 1.314e3, 6.201e2, 2.904e-3, 6.92397e-2), sampledata)
s0 <- apply(G0, 2, sd) # SD of each column

# Start with simulated annealing/stochastic exploration to get initial guess
obj <- function(theta, x) {
  # w1 <- 10 # weight the first 12 moments 10x as much as the others
  # w2 <- 1 # baseline

  G <- reduce_dimension(theta, x)
  G <- sweep(G, 2, s0, "/") # scale each column by its SD
  gbar <- colMeans(G)  # 1 x k

  k <- length(gbar)
  W <- diag(k)  # 1st-stage: identity weighting

  as.numeric(t(gbar) %*% W %*% gbar)
}

# res <- optim(
#   par     = c(-0.05, -3.87, -3.39,  0.56,  4.30, -0.69, -0.64,  0.23,  2.66, -0.18, -1.67, -2.08, 5.18, -0.47, -0.47, -7.55,  2.65),
#   fn      = obj,
#   x       = sampledata,
#   method  = "SANN",
#   control = list(maxit = 3000, temp = 150, tmax = 25, trace = 2) # in practice, doesn't need more than ~2000 iterations
# )

# change gamma1 to -4.3 for both theta values (close to 0 but negative), gamma2 to -1 (should be 731k)
t0 <- c(-1.07, -0.7, # utility shifter
        7.28, 1.203, # gamma 1
        -8.40, 6.57, # gamma 2
        -0.95, 1.55, # gamma 3
        -0.696, -0.053, -0.095, -0.347, # gamma 4
        6.219, -1.889, -2.716,  #gamma 5 - gamma 7
        -3, -1.598)

# Now refine and get standard errors (hopefully)
myouts <- gmm(reduce_dimension, # moment_function # returns *individual-level moment contributions*,
              x=sampledata,
              # wmatrix="ident", # to get a first step estimate
              onlyCoefficients = TRUE, # start here to get to correct parameters before trying to invert (takes weeks to get standard errors)
              t0 = t0, # res$par, # c(84, 79, 3000, 4000, 60, 30, 1, 1, -3.5, 0, 10, 10), # starting values for just the main utility moments for now
              crit=1e-10, tol=1e-10, # (default = 1e-7)
              control = list(parscale=scale_ratio, trace=2)) # to handle differently scaled parameters 

save.image("RcppModel_GMM_Estimation_GMM_20260223.RData") # save progress
beep(sound=0) # random noise to indicate that code has finished
################################################################################