// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Utility functions
inline double logit_inv(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

// lambdahat = subjective false positive rate f = P(z=1 | no cancer)
// rhohat    = subjective true positive rate    = P(z=1 | cancer)

// [[Rcpp::export]]
double post_b_given_z0(double b, double rhohat, double lambdahat) {
  // P(cancer|z=0) = (1-rhohat)*b / [(1-rhohat)*b + (1-lambdahat)*(1-b)]
  double num = (1.0 - rhohat) * b;
  double den = num + (1.0 - lambdahat) * (1.0 - b);
  double val = num / den;
  return std::max(0.001, std::min(val, 0.99));
}

// We actually don't need this, given that we assume confirmatory testing happens
//double post_b_given_z1(double b, double rhohat, double lambdahat) {
//  // P(cancer|z=1) = rhohat*b / [rhohat*b + fpr*(1-b)]
//  double num = rhohat * b;
//  double den = num + lambdahat * (1.0 - b);
//  double val = num / den;
//  return std::max(0.001, std::min(val, 0.99));
//}

// [[Rcpp::export]]
NumericVector logit_inv(const NumericVector& x) {
  // vectorized: 1 / (1 + exp(-x))
  return 1.0 / (1.0 + exp(-x));
}

// internal helper: can use pointers
inline double utility(int s,
               double x,
               double h,
               double z,
               int theta, // note: theta in {1,2,3}
               int d, 
               int age,
               double sympt, // symptoms in {0,1}
               double w,
               double ps,
               double px,
               double py,
               double rhohat,                // subjective  (P(z=1|cancer))
               double lambdahat,             // fpr (P(z=1|no cancer))
               double rho,                  // objective  (P(z=1|cancer))  
               double lambda,               // objective fpr (P(z=1|no cancer))
               const double* gamma,
               const double* gamma0,
               const double* gamma1,
               const double* gamma2,
               const double* gamma3,
               double gamma4, // this one is given as type-specific, not the whole vector
               double gamma5, 
               double gamma6,
               double gamma7,
               double bt) {
  // pick which utility params to use (te = 0 or 1 for C++ 0-based)
  // use the late stage parameters only if late stage *and* diagnosed 
  const int te = ((theta < 3 || d == 0) ? 0 : 1);
  
  // eligibility: age at least 50, h>19, s==1 (or quit within 15 years)
  const int elig_i = (age > 49 && age < 120 && h > 19) ? 1 : 0;
  
  // consumption
  const double cons = w
  - ps * s
  - x * (1 - elig_i) * px // no cost for initial screening if eligible  
  - x * z * (px + py * (theta != 1)); // follow-up screening costs if initial test is positive 
  
  // penalize non-positive, NaN or infinite consumption
  if (cons <= 0.0 || std::isnan(cons) || std::isinf(cons)) {
    return -1e10;
  }
  
  // CRRA utility
  const double g0 = gamma0[te];
  double uc;
  if (std::fabs(g0 - 1.0) < 1e-12) {
    uc = std::log(cons);
  } else {
    uc = std::pow(cons, 1.0 - g0) / (1.0 - g0);
  }
  
  // full utility
  const double p_pos = (theta == 1) ? lambda : rho;
  
  // baseline + smoking interaction terms
  const double hs = h * s;
  const double u_base = uc
  + gamma[te]
  + gamma1[te] * s
  + gamma2[te] * hs
  - gamma3[te] * h * hs
  - gamma7 * sympt
  - gamma6 * bt * 100.0; // incur disutility from symptoms + having cancer/beliefs in all cases
  
  // if late stage and diagnosed, no screening disutility/cost terms (matches original logic)
  if (theta == 3 && d == 1) {
    return u_base;
  }
  
  // screening and belief-related terms
  return u_base
  - x * gamma4
  - gamma5 * x * ((1 - elig_i) + p_pos) * px;
}

// State-transition probabilities
inline void theta_transition(double&p0, double&p1, double&p2, 
                             int theta, int h, int s, int age, int y,
                              const double* alpha,
                              const double* zeta) { // const NumericVector& delta,
  // output: (p0,p1,p2) corresponding to Pr(theta'=1,2,3)
  if (theta == 1) { // note: theta \in {1,2,3}
    const double sum1 = alpha[0] + alpha[1]*h/10 + alpha[2]*s + alpha[3]*(age-40)/10;
    const double q = logit_inv(sum1);
    p0 = 1.0 - q;
    p1 = q;
    p2 = 0.0;
  } else if (theta == 3) { 
    p0 = 0.0; p1 = 0.0; p2 = 1.0;
  }
}

inline void theta_transition2(double&p0, double&p1, double&p2, 
                             int tau, int y,
                             const double* alpha,
                             const double* zeta) { // const NumericVector& delta,
  // output: (p0,p1,p2) corresponding to Pr(theta'=1,2,3)
    const double xb2 = zeta[0] + zeta[1] * tau + zeta[2] * y + zeta[3] * tau * y; 
    const double q2 = logit_inv(xb2); 
    
    // no recovery:  double xb1 = delta[0] + delta[1]*h/10 + delta[2]*s + delta[3]*(age-40)/10 + delta[4]*y;
    // const double xb2 = zeta[0] + zeta[1]*h/10 + zeta[2]*s + zeta[3]*(age-40)/10 + zeta[4]*y; // progression to late stage cancer
    // no recovery: double p0 = logit_inv(xb1);
    // const double q2 = logit_inv(xb2);
    // no recovery: p[0] = p0;
    p0 = 0.0; // no recovery from early stage to no stage cancer
    p1 = std::max(0.0, 1.0 - q2); // - p0 if you put recovery back in
    p2 = 1.0-p1;
}

inline void phi_transition(double& p0, double&p1, 
                           int phi, int h, int s, int age,
                           const double* psi) {
  // output: (p0,p1) corresponding to Pr(phi'=1,2)
  double xb = psi[0] + psi[1]*s + psi[2]*(age-40)/10 + psi[3]*(age-40)/10*s;
  if (phi == 1) {
    const double q = logit_inv(xb);
    p0 = 1 - q;
    p1 = q;
  } else {
    p0 = 0;
    p1 = 1;
  }
}

inline double survival_prob(int theta, int phi, int age,
                                const double* omega) {
  const int th2 = (theta == 2);
  const int th3 = (theta == 3);
  const int ph2 = (phi == 2);
  const double xb = omega[0] + omega[1]*th2 + omega[2]*th3 + omega[3]*ph2 + omega[4]*(age-40)/10;
  const double q = logit_inv(xb); // death prob 
  return 1.0 - q; // survival prob
}

//inline void symptoms_prob(double& p0, double&p1,
//                            int theta, int h, int s, int age, int x, double rho, int d,
//                            const double* kappa) { 
//  if (theta < 3 || d == 1) { // no symptomatic diagnosis unless you're in late-stage cancer
//    p0 = 1.0; 
//    p1 = 0.0; 
//  } else if (x == 1 && theta == 1) { // if screened, then positive only with objective true positive probability, no symptomatic diagnosis
//    p0 = 1.0; 
//    p1 = 0.0; // note: false positive screenings don't convert into dx
//  } else if (x == 1 && theta == 2) { // if screened, then positive only with objective true positive probability, no symptomatic diagnosis
//    p0 = 1.0-rho; 
//    p1 = rho; // probability of true positive (which will be confirmed)
//  } else { // if you have late stage cancer and you aren't diagnosed and you haven't screened, then you may have symptoms 
//    const double xb = kappa[0] + kappa[1]*h/10 + kappa[2]*s + kappa[3]*(age-40)/10;
//    const double q1 = logit_inv(xb); 
//    p0 = 1.0-q1; 
//    p1 = q1; // prob of having symptoms 
//  }
//}

// [[Rcpp::export]]
List solve_model(int T, int tmin,
                 const Rcpp::NumericVector& b_grid,
                 const IntegerVector& h_grid,
                 double beta,
                 const Rcpp::NumericVector& gamma,
                 const Rcpp::NumericVector& gamma0,
                 const Rcpp::NumericVector& gamma1,
                 const Rcpp::NumericVector& gamma2,
                 const Rcpp::NumericVector& gamma3,
                 const Rcpp::NumericVector& gamma4,
                 double gamma5,
                 double gamma6,
                 double gamma7,
                 double ps,
                 double px,
                 double py,
                 double w,
                 double rhohat,
                 double lambdahat,
                 double rho,
                 double lambda,
                 const Rcpp::NumericVector& alpha,
                 const Rcpp::NumericVector& zeta,
                 const Rcpp::NumericVector& psi,
                 const Rcpp::NumericVector& omega) { // const NumericVector& delta,
  
  int n_b = b_grid.size();
      // Rcpp::Rcout << "  nb=" << n_b << std::endl;
  //if (n_b != 101) {
  //  Rcpp::stop("With current belief indexing (round(b*100)), b_grid must have length 101 (0.00..1.00 by 0.01).");
  //}
  int n_h = h_grid.size();
  int n_theta = 3;
  
  // Raw pointers to coefficient vectors (avoid repeated operator[] overhead in hot loops)
  const double* alpha_p = REAL(alpha);
  const double* zeta_p  = REAL(zeta);
  const double* psi_p   = REAL(psi);
  const double* omega_p = REAL(omega);
  const double* gamma_p  = REAL(gamma);
  const double* gamma0_p = REAL(gamma0);
  const double* gamma1_p = REAL(gamma1);
  const double* gamma2_p = REAL(gamma2);
  const double* gamma3_p = REAL(gamma3);
  const double* gamma4_p = REAL(gamma4);
  const double* b_p = REAL(b_grid);
  const int* h_p = INTEGER(h_grid);
  int tau_max = 14; 
  
  // 7D array for V: dim = [n_b, n_h, 3, 2, T+1,d=diagnosed, type=4]
  int D1 = n_b, D2 = n_h, D3 = 3, D4 = 2, D5 = T + 1, D6 = 2, D7 = 4;
  int totV = D1 * D2 * D3 * D4 * D5 * D6 * D7;
  Rcpp::NumericVector V(totV);
  Rcpp::IntegerVector dimV = Rcpp::IntegerVector::create(D1, D2, D3, D4, D5, D6, D7);
  V.attr("dim") = dimV;
  
  // 8D array for policy: dim = [n_b, n_h, 3, 2, 2, T+1, d=diagnosed, type=4]
  int D8 = 2;
  int totP = D1 * D2 * D3 * D4 * D8 * D5 * D6 * D7;
  
  Rcpp::IntegerVector policy(totP);
  Rcpp::IntegerVector dimP = Rcpp::IntegerVector::create(D1, D2, D3, D4, D8, D5, D6, D7);
  policy.attr("dim") = dimP;

  //if (b_grid.size() != D1) {
  //  Rcpp::stop("b_grid length must match policy dim[0] (n_b).");
  //}
  //if (D1 != 101) {
  //  Rcpp::stop("With current belief indexing (round(b*100)), policy must use n_b=101.");
  //}
  
  // ARRAYS FOR WHEN THETA = 2 (BECAUSE THEN PROGRESSION TO THETA=3 DEPENDS ON TIME SINCE DIAGNOSIS)
  // 7D array for V2: dim = [n_b, n_h, 2, T+1,d=diagnosed, tau=1:15, type=4]
  int Dtau = 15; 
  Rcpp::NumericVector V2(D1 * D2 * D4 * D5 * D6 * Dtau * D7);  
  Rcpp::IntegerVector dimV2 = Rcpp::IntegerVector::create(D1, D2, D4, D5, D6, Dtau, D7);
  V2.attr("dim") = dimV2;
  
  // 8D array for policy: dim = [n_b, n_h, 2, 2, T+1, d=diagnosed, tau=1:15, type=4]
  Rcpp::IntegerVector policy2(D1 * D2 * D4 * D8 * D5 * D6 * Dtau * D7);
  Rcpp::IntegerVector dimP2 = Rcpp::IntegerVector::create(D1, D2, D4, D8, D5, D6, Dtau, D7);
  policy2.attr("dim") = dimP2;
  
  // Helpers to index flattened arrays
  auto idxV = [&](int bi, int hi, int th1, int ph1, int t1, int d, int type) -> int {
    const int th = th1 - 1;  // 0..2
    const int ph = ph1 - 1;  // 0..1
    const int tt = t1  - 1;  // 0..T
    return bi + D1 * (hi + D2 * (th + D3 * (ph + D4 * (tt + D5 * (d + D6 * type)))));
  };
  auto idxP = [&](int bi, int hi, int th1, int ph1, int a, int t1, int d, int type) -> int {
    const int th = th1 - 1;
    const int ph = ph1 - 1;
    const int tt = t1  - 1;
    return bi + D1 * (hi + D2 * (th + D3 * (ph + D4 * (a + D8 * (tt + D5 * (d + D6 * type))))));
  };
  auto idxV2 = [&](int bi, int hi, int ph1, int t1, int d, int tau, int type) -> int {
    const int ph = ph1 - 1;
    const int tt = t1  - 1;
    return bi + D1 * (hi + D2 * (ph + D4 * (tt + D5 * (d + D6 * (tau + Dtau * type)))));
  };
  auto idxP2 = [&](int bi, int hi, int ph1, int a, int t1, int d, int tau, int type) -> int {
    const int ph = ph1 - 1;
    const int tt = t1  - 1;
    return bi + D1 * (hi + D2 * (ph + D4 * (a + D8 * (tt + D5 * (d + D6 * (tau + Dtau * type))))));
  };
  
  auto b_to_idx = [&](double b){
    b = std::min(1.0, std::max(0.0, b));
    return (int)std::round(b * 100.0);
  };
  
  auto round2 = [&](double x){
    return std::round(x * 100.0) / 100.0;
  };
  
  // Main loop
  // loop goes through T, 
  for (int t = T; t >= tmin; --t) {
    if (tmin < 1) Rcpp::stop("tmin must be >= 1 because DP index uses (t-1).");
    if (T < tmin) Rcpp::stop("T must be >= tmin.");
    
    // here we can calculate all needed survival probabilities 
    // arguments are (theta, phi, t)
    const double p_d11 = survival_prob(1, 1, t+1, omega_p);
    const double p_d12 = survival_prob(1, 2, t+1, omega_p);
    const double p_d21 = survival_prob(2, 1, t+1, omega_p);
    const double p_d22 = survival_prob(2, 2, t+1, omega_p);
    const double p_d31 = survival_prob(3, 1, t+1, omega_p); // surviving without chronic condition
    const double p_d32 = survival_prob(3, 2, t+1, omega_p); // surviving with chronic condition
    
    double sympt_draw = R::runif(0.0, 1.0); // initial draw for symptoms in each period
    int sympt = 0; // no symptoms is the default
    
    for (int theta = 1; theta <= n_theta; ++theta) { // theta in {1,2,3}
      
      // note: DO THIS SEPARATELY FOR EACH OF THETA={1,3} AND THETA=2 BECAUSE OF PROGRESSION
      if (theta == 2) {
        for (int tau=0; tau<=tau_max; ++tau) { // loops through 0-14 years of diagnosis
          double p_theta_20_0, p_theta_20_1, p_theta_20_2; // theta = 2, y = 0 
          double p_theta_21_0, p_theta_21_1, p_theta_21_2; // theta = 2, y = 1
          theta_transition2(p_theta_20_0, p_theta_20_1, p_theta_20_2, tau, 0, alpha_p, zeta_p); // delta from early stage with no treatment (not diagnosed)
          theta_transition2(p_theta_21_0, p_theta_21_1, p_theta_21_2, tau, 1, alpha_p, zeta_p); // delta from early stage with treatment (all diagnosed have treatment)
        #pragma omp parallel for collapse(2) // Parallelize b and h loops
        for (int b_i = 0; b_i < n_b; ++b_i) {
          for (int h_i = 0; h_i < n_h; ++h_i) {
            double bt = b_grid[b_i]; 
            int ht = h_p[h_i]; // note that since h_grid has gap of 1 year, ht = h_i+1 always
            for (int phi = 1; phi <= 2; ++phi) { // phi in {1,2}
              
              for (int d = 0; d <= 1; ++d) { // diagnosed state in {0, 1}
                for (int type = 0; type < 4; ++type) {
                  double best_val = -1e8; // reset best value here 
                  int best_s = 0, best_x = 0; // reset best policy choices here
                  
                  for (int s = 0; s <= 1; ++s) { // smoking decision
                    
                    // here, we can include phi_transition and theta_transition parameters 
                    double p_phi0,p_phi1; // Pr(phi' = 1, 2)
                    phi_transition(p_phi0,p_phi1, phi, ht, s, t, psi_p);
                    
                    double p_theta_10_0, p_theta_10_1, p_theta_10_2; // theta = 1 (incorrect beliefs), y = 0 
                    theta_transition(p_theta_10_0, p_theta_10_1, p_theta_10_2, 1, ht, s, t, 0, alpha_p, zeta_p); // delta from no cancer with no treatment 
                    // note: theta' = 3 always when theta = 3 (no transition probabilities needed)
                    
                    for (int x = 0; x <= 1; ++x) { // screening decision
                      double bt2;
                      // calculate the utility in the given period
                      double u;
                      if (d == 1) { // if already diagnosed with cancer of one stage or another 
                          bt2 = 1; 
                          // early stage cancer → must still screen to know stage, but no false positives now (test will return early stage)
                          u = utility(
                            s, 1, ht, 1, 2, 1, t, 1, // symptoms are 1 
                            w, ps, px, py, rhohat, lambdahat, rho, lambda,
                            gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, 1); // beliefs about cancer are now 1
                      } else if (d == 0) { // if not yet diagnosed, choose whether or not to screen
                        
                        // draw symptoms based on theta -- when theta = 2, symptoms happen with probability 0.064
                        sympt = (sympt_draw < 0.064) ? 1 : 0;
                        
                        if (sympt == 1) { // individual assumes that cancer may be early or late stage with equal probability 
                          bt2 = (0.156 * bt) / (0.156 * bt + 0.033 * (1-bt)); // std::max(0.0, std::min( (0.156 * bt) / (0.156 * bt + 0.033 * (1-bt)), 0.99)); 
                        } else { 
                          bt2 = ((1-0.156) * bt) / ((1-0.156) * bt + (1-0.033) * (1-bt)); // std::max(0.0, std::min( ((1-0.156) * bt) / ((1-0.156) * bt + (1-0.033) * (1-bt)), 0.99));
                        }
                        
                        if (x == 0) {
                          // no screening
                          u = utility(
                            s,       // s
                            0,       // x = 0 → no test
                            ht,      // current health
                            0,       // z=0 (no test result)
                            theta,   // (unknown) disease stage in {1,2,3}
                            0,       // not diagnosed 
                            t,       // age/time
                            sympt,   // symptoms?
                            w, ps, px, py, rhohat, lambdahat, rho, lambda,
                            gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2);  
                        } else {
                          // screening: take expectation over possible outcomes
                          double u11 = utility(s, 1, ht, 1, 1, 0, t, sympt, w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                               gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2); // theta = 1 and z = 1 (false positive)
                          double u01 = utility(s, 1, ht, 0, 1, 0, t, sympt, w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                               gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2); // theta = 1 and z = 0 (true negative)
                          double u02 = utility(s, 1, ht, 0, 2, 0, t, sympt, w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                               gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2);
                          double u12 = utility(s, 1, ht, 1, 2, 0, t, sympt, w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                               gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2);
                          u = (1.0 - bt2) * lambdahat * u11 // using subjective probabilities, and assuming no false "over-staging" so if there is a true positive, state is known
                          + (1.0 - bt2) * (1.0 - lambdahat) * u01 
                          + bt2 * (rhohat * u12 + (1.0-rhohat) * u02); // If you have a new diagnosis theta = 2 here
                        }
                      } // we have now defined utility in all possible states of d, theta, and screening
                      
                      // update the state vector
                      // smoking history update
                      int hi1 = std::min(h_i + s, n_h - 1);   // index update
                      int ht_next = h_p[hi1];             // next level if needed
                      // int ht1    = std::min(ht + s, n_h); // move smoking up one if s == 1, cap smoking history at n_h pack-years
                      int at1    = t + 1;
                      
                      // update beliefs for next period
                      // if d == 1, then beliefs should be automatically 1 (index = 100)
                      
                      // --- x = 0 case ---
                      double bt1_0   = round2(bt2 + (1.0 - bt2) * p_theta_10_1);   
                      // rounded to 2 decimal places to avoid needing the helper above, which is commented out. 
                      int    bt1_0_idx = b_to_idx(bt1_0); // closest_idx(bt1_0);
                      
                      // --- x = 1, z = 0 case ---
                      double btilde  = std::min( post_b_given_z0(bt2, rhohat, lambdahat), 0.99 ); // subjective posterior belief of Pr(cancer)
                      double bt1_10  = round2(btilde + (1.0 - btilde) * p_theta_10_1); // weighted average of posterior and transition probability? 
                      int    bt1_10_idx = b_to_idx(bt1_10); // closest_idx(bt1_10);
                      
                      // --- x = 1, z = 1, theta = 1 case ---
                      double bt1_111 = round2(p_theta_10_1); // std::round(100.00*p_theta_10_1)/100.00;
                      int    bt1_111_idx = b_to_idx(bt1_111); // bt1_111*100.00; // closest_idx(bt1_111);
                      
                      // --- x = 1, z = 1, theta = 2 case ---
                      double bt1_112 = round2(p_theta_21_1); // std::round(100.00*p_theta_21_1)/100.00;
                      int    bt1_112_idx = b_to_idx(bt1_112); // bt1_112*100.00; // closest_idx(bt1_112);
                      
                      // Next period value function
                      double val_future = 0.0; 
                      
                      // 1. Terminal period
                      if (t == T) {
                        val_future = 0.0; // confirmed that val_future always works at T=100
                        
                      } else {
                        
                        // 3. Early stage cancer
                          
                          if (d == 1) { 
                            // note: here screening = 1 and y = 1 by default
                            // integrate over next‐period θ∈{2,3} and φ∈{1,2} with d=1 absorbing
                            val_future =
                              p_theta_21_1*p_phi0*p_d21*V2[idxV2(100, ht_next, 1, at1, 1, tau,type)] +
                              p_theta_21_1*p_phi1*p_d22*V2[idxV2(100, ht_next, 2, at1, 1, tau,type)] +
                              p_theta_21_2*p_phi0*p_d31*V[idxV(100, ht_next, 3, 1, at1, 1,type)] +
                              p_theta_21_2*p_phi1*p_d32*V[idxV(100, ht_next, 3, 2, at1, 1,type)];
                          } else if (d == 0) {
                            // here, screening is a choice, and there is no treatment 
                            if (x == 0) { // if no screening today, then cancer will not be detected
                              double part1 =
                                p_theta_20_1*p_phi0*p_d21*V2[idxV2(bt1_0_idx, ht_next, 1, at1, 0, tau,type)] +
                                p_theta_20_1*p_phi1*p_d22*V2[idxV2(bt1_0_idx, ht_next, 2, at1, 0, tau,type)] +
                                p_theta_20_2*p_phi0*p_d31*V[idxV(bt1_0_idx, ht_next, 3, 1, at1, 0,type)] +
                                p_theta_20_2*p_phi1*p_d32*V[idxV(bt1_0_idx, ht_next, 3, 2, at1, 0,type)]; // value if undetected
                              
                              double part2 =
                                p_theta_20_1*p_phi0*p_d21*V2[idxV2(100, ht_next, 1, at1, 1, tau,type)] +
                                p_theta_20_1*p_phi1*p_d22*V2[idxV2(100, ht_next, 2, at1, 1, tau,type)] +
                                p_theta_20_2*p_phi0*p_d31*V[idxV(100, ht_next, 3, 1, at1, 1,type)] +
                                p_theta_20_2*p_phi1*p_d32*V[idxV(100, ht_next, 3, 2, at1, 1,type)]; // value if detected
                              
                              val_future = (1.0 - bt2) * part1 + bt2 * part2;
                            } else if (x == 1) { // screening when you have cancer 
                              // (bt) cancer × two screening-outcome branches
                              double hasCancer_noDetect =
                                p_theta_20_1*p_phi0*p_d21*V2[idxV2(bt1_10_idx, ht_next, 1, at1, 0, tau,type)] +
                                p_theta_20_1*p_phi1*p_d22*V2[idxV2(bt1_10_idx, ht_next, 2, at1, 0, tau,type)] +
                                p_theta_20_2*p_phi0*p_d31*V[idxV(bt1_10_idx, ht_next, 3, 1, at1, 0,type)] +
                                p_theta_20_2*p_phi1*p_d32*V[idxV(bt1_10_idx, ht_next, 3, 2, at1, 0,type)];
                              
                              double hasCancer_detect =
                                p_theta_21_1*p_phi0*p_d21*V2[idxV2(100, ht_next, 1, at1, 1, tau,type)] +
                                p_theta_21_1*p_phi1*p_d22*V2[idxV2(100, ht_next, 2, at1, 1, tau,type)] +
                                p_theta_21_2*p_phi0*p_d31*V[idxV(100, ht_next, 3, 1, at1, 1,type)] +
                                p_theta_21_2*p_phi1*p_d32*V[idxV(100, ht_next, 3, 2, at1, 1,type)];
                              
                              // NOTE: subjective weighting are: with 1-bt, then I have cancer but I don't think I do (so it is undetected), with bt, then I think I do so I screen and it is detected with some probability 
                              val_future = (1.0-bt2) * hasCancer_noDetect + bt2*((1.0 - rhohat)     * hasCancer_noDetect + rhohat   * hasCancer_detect);
                            }
                          }
                      } // this concludes calculation of val_future
                      
                      double V_try = u + beta * val_future;
                      if (V_try > best_val) {
                        best_val = V_try;
                        best_s = s;
                        best_x = x;
                      } 
                    } // close off choice of x 
                  } // close off choice of s
                  
                  // print message for debugging purposes
                  // Rcpp::Rcout << "best value: " << best_val
                  //             << std::endl;
                  
                  V2[idxV2(b_i, h_i, phi, t, d, tau,type)] = best_val; // note that theta, psi, and t are scooted back in idxV by one since cpp indexes starting at 0
                  policy2[idxP2(b_i, h_i, phi, 0, t, d, tau,type)] = best_s;
                  policy2[idxP2(b_i, h_i, phi, 1, t, d, tau,type)] = best_x;
                  
                } // close off loop over type
              } // close off diagnosis (d) loop
            } // close off phi loop
          } // close off h loop
        } // close off b loop
        } // close off tau loop
      } else { // switch to theta \in {1, 3}
        
        double p_theta_20_0, p_theta_20_1, p_theta_20_2; // theta = 2, y = 0 
        double p_theta_21_0, p_theta_21_1, p_theta_21_2; // theta = 2, y = 1
        theta_transition2(p_theta_20_0, p_theta_20_1, p_theta_20_2, 0, 0, alpha_p, zeta_p); // delta from early stage with no treatment (not diagnosed)
        theta_transition2(p_theta_21_0, p_theta_21_1, p_theta_21_2, 0, 1, alpha_p, zeta_p); // delta from early stage with treatment (all diagnosed have treatment)
      #pragma omp parallel for collapse(2) // Parallelize b and h loops
      for (int b_i = 0; b_i < n_b; ++b_i) {
        for (int h_i = 0; h_i < n_h; ++h_i) {
          double bt = b_p[b_i]; 
          int ht = h_p[h_i]; // note that since h_grid has gap of 1 year, ht = h_i+1 always
          for (int phi = 1; phi <= 2; ++phi) { // phi in {1,2}
            for (int d = 0; d <= 1; ++d) { // diagnosed state in {0, 1}
              for (int type = 0; type < 4; ++type) {
                  double best_val = -1e8; // reset best value here 
                  int best_s = 0, best_x = 0; // reset best policy choices here
                  
                  for (int s = 0; s <= 1; ++s) { // smoking decision
                    
                    // here, we can include phi_transition and theta_transition parameters 
                    double p_phi0,p_phi1; // Pr(phi' = 1, 2)
                    phi_transition(p_phi0,p_phi1, phi, ht, s, t, psi_p);
                    
                    double p_theta_10_0, p_theta_10_1, p_theta_10_2; // theta = 1, y = 0
                    theta_transition(p_theta_10_0, p_theta_10_1, p_theta_10_2, 1, ht, s, t, 0, alpha_p, zeta_p); // delta from no cancer with no treatment 
                    // note: theta' = 3 always when theta = 3 (no transition probabilities needed)
                    
                    for (int x = 0; x <= 1; ++x) { // screening decision 
                      // calculate the utility in the given period
                          double u;
                          double bt2;
                          if (d == 1) { // if already diagnosed with cancer of one stage or another 
                              bt2 = 1; 
                              // already diagnosed → late stage (does not need screening)
                              u = utility(
                                s, 0, ht, 0, 3, 1, t,1, // always has symptoms 
                                w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, 1); // beliefs about cancer are now 1
                            
                          } else if (d == 0) { // if not yet diagnosed, choose whether or not to screen
                            
                            // draw symptoms based on theta -- when theta = 1, symptoms happen with probability 0.033, when theta = 3, probability is 0.246
                            if (theta == 1) { 
                              sympt = (sympt_draw < 0.033) ? 1 : 0;
                            }else { 
                              sympt = (sympt_draw < 0.246) ? 1 : 0; 
                            }
                            if (sympt == 1) { // individual assumes that cancer may be early or late stage with equal probability 
                              bt2 = (0.156 * bt) / (0.156 * bt + 0.033 * (1-bt)); // std::max(0.0, std::min( (0.156 * bt) / (0.156 * bt + 0.033 * (1-bt)), 0.99)); 
                            } else { 
                              bt2 = ((1-0.156) * bt) / ((1-0.156) * bt + (1-0.033) * (1-bt)); // std::max(0.0, std::min( ((1-0.156) * bt) / ((1-0.156) * bt + (1-0.033) * (1-bt)), 0.99));
                            }
                            
                            if (x == 0) {
                              // no screening
                              u = utility(
                                s,       // s
                                0,       // x = 0 → no test
                                ht,      // current health
                                0,       // z=0 (no test result)
                                theta,   // (unknown) disease stage in {1,2,3}
                                0,       // not diagnosed 
                                t,       // age/time
                                sympt,   // symptoms
                                w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2);  
                            } else {
                              // screening: take expectation over possible outcomes
                              double u11 = utility(s, 1, ht, 1, 1, 0, t, sympt, w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                                   gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2); // theta = 1 and z = 1 (false positive)
                              double u01 = utility(s, 1, ht, 0, 1, 0, t, sympt, w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                                   gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2); // theta = 1 and z = 0 (true negative)
                              double u03 = utility(s, 1, ht, 0, 3, 0, t, sympt, w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                                   gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2);
                              double u13 = utility(s, 1, ht, 1, 3, 0, t, sympt, w, ps, px, py, rhohat, lambdahat, rho, lambda,
                                                   gamma_p, gamma0_p, gamma1_p, gamma2_p, gamma3_p, gamma4_p[type], gamma5, gamma6, gamma7, bt2);
                              u = (1.0 - bt2) * lambdahat * u11 // using subjective probabilities, and assuming no false "over-staging" so if there is a true positive, state is known
                              + (1.0 - bt2) * (1.0 - lambdahat) * u01 
                              + bt2 * (rhohat * u13 + (1.0-rhohat) * u03); // if you have a new diagnosis here then theta = 3
                            }
                          } // we have now defined utility in all possible states of d, theta, and screening
                          
                          // update the state vector
                          // smoking history update
                          int hi1 = std::min(h_i + s, n_h - 1);   // index update
                          int ht_next = h_p[hi1];             // next level if needed
                          // int ht1    = std::min(ht + s, n_h); // move smoking up one if s == 1, cap smoking history at n_h pack-years
                          int at1    = t + 1;
                          
                          // helper to find nearest index in b_grid
                          //auto closest_idx = [&](double val) {
                          //  int best = 0;
                          //  double best_diff = std::abs(b_grid[0] - val);
                          //  for (int i = 1; i < b_grid.size(); ++i) {
                          //    double d = std::abs(b_grid[i] - val);
                          //    if (d < best_diff) { best_diff = d; best = i; }
                          //  }
                          //  return best;
                          //};
                          
                          // update beliefs for next period
                            // if d == 1, then beliefs should be automatically 1 (index = 100)
                            
                            // --- x = 0 case ---
                            double bt1_0   = round2(bt2 + (1.0 - bt2) * p_theta_10_1);   
                              // rounded to 2 decimal places to avoid needing the helper above, which is commented out. 
                            int    bt1_0_idx = b_to_idx(bt1_0); // closest_idx(bt1_0);
                            
                            // --- x = 1, z = 0 case ---
                            double btilde  = std::min( post_b_given_z0(bt2, rhohat, lambdahat), 0.99 ); // subjective posterior belief of Pr(cancer)
                            double bt1_10  = round2(btilde + (1.0 - btilde) * p_theta_10_1); // weighted average of posterior and transition probability? 
                            int    bt1_10_idx = b_to_idx(bt1_10); // closest_idx(bt1_10);
                            
                            // --- x = 1, z = 1, theta = 1 case ---
                            double bt1_111 = round2(p_theta_10_1); // std::round(100.00*p_theta_10_1)/100.00;
                            int    bt1_111_idx = b_to_idx(bt1_111); // bt1_111*100.00; // closest_idx(bt1_111);
                            
                            // --- x = 1, z = 1, theta = 2 case ---
                            double bt1_112 = round2(p_theta_21_1); // std::round(100.00*p_theta_21_1)/100.00;
                            int    bt1_112_idx = b_to_idx(bt1_112); // bt1_112*100.00; // closest_idx(bt1_112);
                          
                          // Next period value function
                          double val_future = 0.0; 
                          
                          // 1. Terminal period
                          if (t == T) {
                            val_future = 0.0; // confirmed that val_future always works at T=100
                            
                          } else {
                            
                          // 2. Late‐stage cancer: 
                          if (theta == 3) {
                            if (d == 1) { // only beliefs slot = 101 *if diagnosed* (0‐based index 100)
                            // V[belief=101, ht1, theta=3, z, at1]
                            val_future =
                              p_phi0 * p_d31 * V[idxV(100, ht_next, 3, 1, at1, 1, type)]+ // hi1
                              p_phi1 * p_d32 * V[idxV(100, ht_next, 3, 2, at1, 1, type)]; // hi1
                            // confirmed that these are constructed correctly
                            } else if (d == 0) { // if not diagnosed
                              if (x == 0) { 
                                
                                // integrate over next‐period θ∈{3} (absorbing state), d∈{0, 1} and φ∈{1,2} (with survival)
                                // future val is broken up into diagnosed + not diagnosed in next period, with subjective beliefs bt
                                val_future = (1.0-bt2) * (
                                    p_phi0*p_d31*V[idxV(bt1_0_idx, ht_next, 3, 1, at1, 0, type)] + // not diagnosed next period
                                    p_phi1*p_d32*V[idxV(bt1_0_idx, ht_next, 3, 2, at1, 0, type)]
                                  ) + 
                                  bt2 * (p_phi0*p_d31*V[idxV(100, ht_next, 3, 1, at1, 1, type)] + // diagnosed next period
                                    p_phi1*p_d32*V[idxV(100, ht_next, 3, 2, at1, 1, type)]);
                                
                              } else if (x == 1) { 
                                // with cancer and screening, then there are only two screening-outcome branches
                                double hasCancer_noDetect =
                                  p_phi0*p_d31*V[idxV(bt1_10_idx, ht_next, 3, 1, at1, 0, type)] +
                                  p_phi1*p_d32*V[idxV(bt1_10_idx, ht_next, 3, 2, at1, 0, type)];
                                
                                double hasCancer_detect =
                                  p_phi0*p_d31*V[idxV(100, ht_next, 3, 1, at1, 1, type)] +
                                  p_phi1*p_d32*V[idxV(100, ht_next, 3, 2, at1, 1, type)];
                                
                                // NOTE: subjective weighting are: with 1-bt, then I have cancer but I don't think I do (so it is undetected), with bt, then I think I do so I screen and it is detected with some probability 
                                val_future = (1.0-bt2)*hasCancer_noDetect + bt2*((1.0 - rhohat)     * hasCancer_noDetect + rhohat   * hasCancer_detect);
                              }
                            }
                            // 4. No cancer 
                          } else if (theta == 1) {
                            if (x == 0) { // no screening
                              // integrate over next‐period θ∈{1,2,3} and φ∈{1,2}
                              double part1 =
                                p_theta_10_0*p_phi0*p_d11*V[idxV(bt1_0_idx, ht_next, 1, 1, at1, 0, type)] +
                                p_theta_10_0*p_phi1*p_d12*V[idxV(bt1_0_idx, ht_next, 1, 2, at1, 0, type)]; // no cancer next period
                              
                              double part2 =
                                p_theta_10_1*p_phi0*p_d21*V2[idxV2(bt1_0_idx, ht_next, 1, at1, 0, 0, type)] + // note that tau = 0
                                p_theta_10_1*p_phi1*p_d22*V2[idxV2(bt1_0_idx, ht_next, 2, at1, 0, 0, type)] +
                                p_theta_10_2*p_phi0*p_d31*V[idxV(bt1_0_idx, ht_next, 3, 1, at1, 0, type)] +
                                p_theta_10_2*p_phi1*p_d32*V[idxV(bt1_0_idx, ht_next, 3, 2, at1, 0, type)]; // undetected cancer 
                              
                              val_future = (1.0 - bt) * part1 + bt * part2; // weighting of true undetected cancer (with bt) and no cancer next period 
                            } else if (x == 1) { 
                              // if there is no cancer and you screen, you either have noCancer_detect (false positive) or noCancer_noDetect (true negative)
                              // however, diagnosis state doesn't update, so d=0 still. Future valuations are just based on how theta and phi evolve (costs of false positive are only in the current period)
                              val_future = 
                                (1.0-bt2) * (p_theta_10_0*p_phi0*p_d11*V[idxV(bt1_10_idx, ht_next, 1, 1, at1, 0, type)] +
                                p_theta_10_0*p_phi1*p_d12*V[idxV(bt1_10_idx, ht_next, 1, 2, at1, 0, type)]) +
                                bt2 * (p_theta_10_1*p_phi0*p_d21*V2[idxV2(bt1_10_idx, ht_next, 1, at1, 0, 0, type)] +
                                p_theta_10_1*p_phi1*p_d22*V2[idxV2(bt1_10_idx, ht_next, 2, at1, 0, 0, type)] +
                                p_theta_10_2*p_phi0*p_d31*V[idxV(bt1_10_idx, ht_next, 3, 1, at1, 0, type)] +
                                p_theta_10_2*p_phi1*p_d32*V[idxV(bt1_10_idx, ht_next, 3, 2, at1, 0, type)]);
                            }
                          }
                          } // this concludes calculation of val_future
                          
                      double V_try = u + beta * val_future;
                      if (V_try > best_val) {
                        best_val = V_try;
                        best_s = s;
                        best_x = x;
                      } 
                    } // close off choice of x 
                  } // close off choice of s
                  
                  // print message for debugging purposes
                  // Rcpp::Rcout << "best value: " << best_val
                  //             << std::endl;
                  
                  V[idxV(b_i, h_i, theta, phi, t, d, type)] = best_val; // note that theta, psi, and t are scooted back in idxV by one since cpp indexes starting at 0
                  policy[idxP(b_i, h_i, theta, phi, 0, t, d, type)] = best_s;
                  policy[idxP(b_i, h_i, theta, phi, 1, t, d, type)] = best_x;
                
              } // close off type loop
            } // close off diagnosis (d) loop
          } // close off phi loop
        } // close off h loop
      } // close off b loop
      } // close off separate loop for theta \in {1, 3}
    } // close off theta loop
    //Rcpp::Rcout << "Value: " << V[idxV(10, 10, 1, 1, t)]
    //            << std::endl;
  } // close off T loop
  
  return List::create(
    _["V"] = V,
    _["policy"] = policy,
    _["V2"] = V2,
    _["policy2"] = policy2
  );
}

// NOW INCORPORATING A FORWARD SIMULATION OF N INDIVIDUALS FROM AGE 40 
// [[Rcpp::export]]
Rcpp::DataFrame simulate_cohort_cpp(
    int N,                        // number of individuals
    int start_age,                // e.g. 40
    int max_age,                  // e.g. 100
    const NumericVector& b_grid,  // belief grid, e.g. seq(0, 1, 0.01)
    const IntegerVector& h_grid,  // history grid, e.g. 0:40
    const IntegerVector& policy,  // 6D policy array from solve_model
    const IntegerVector& policy2,  // policy specifically for theta =2
    const NumericVector& alpha,  // const NumericVector& delta, // omitted for now
    const NumericVector& zeta,
    const NumericVector& psi,
    const NumericVector& omega,  
    double rhohat,                // subjective (P(z=1|cancer))
    double lambdahat,             // subjective false positive  (P(z=1|no cancer))
    double rho,                // objective (P(z=1|cancer))
    double lambda,             // objective false positive (P(z=1|no cancer))
    const NumericVector& b0,      // initial beliefs (length N)
    const IntegerVector& h0,      // initial histories (length N)
    const IntegerVector& theta0,  // initial cancer state (1,2,3) (length N)
    const IntegerVector& phi0,     // initial chronic state (1,2) (length N)
    const IntegerVector& type_vec     // initial type (optimism x stigma)
) {
  RNGScope scope;
  
  // Raw pointers to coefficient vectors (avoid Rcpp overhead in loops)
  const double* alpha_p = REAL(alpha);
  const double* zeta_p  = REAL(zeta);
  const double* psi_p   = REAL(psi);
  const double* omega_p = REAL(omega);
  
  // Dimensions of policy array
  IntegerVector dimP = policy.attr("dim"); // [D1, D2, 3, 2, 2, T+1, d=2, type=4]
  if (dimP.size() != 7 && dimP.size() != 8) {
    Rcpp::stop("policy must have a dim attribute of length 7 or 8; got %d", dimP.size());
  }
  int D1 = dimP[0]; // n_b
  int D2 = dimP[1]; // n_h
  int D3 = dimP[2]; // 3 (theta)
  int D4 = dimP[3]; // 2 (phi)
  int D5 = dimP[4]; // 2 (actions: s,x)
  int D6 = dimP[5]; // T+1 (time index)
  int D7 = dimP[6]; // 2 (diagnosis state)
  // int D8 = dimP[7]; // type (optimism x stigma)
  
  // Dimensions of policy array
  IntegerVector dimP2 = policy2.attr("dim"); // [D1, D2, 2, 2, T+1, d=2, tau, type]
  if (dimP2.size() != 8) {
    Rcpp::stop("policy2 must have a dim attribute of length 8; got %d", dimP2.size());
  }
  int D21 = dimP2[0]; // n_b
  int D22 = dimP2[1]; // n_h
  int D23 = dimP2[2]; // 2 (phi)
  int D24 = dimP2[3]; // 2 (actions: s,x)
  int D25 = dimP2[4]; // T+1 (time index)
  int D26 = dimP2[5]; // 2 (diagnosis state)
  int D27 = dimP2[6]; // tau 
  // int D28 = dimP2[7]; // type (optimism x stigma)
  
  // Helper for policy index (copying structure from solve_model)
  auto idxP = [&](int bi, int hi, int th, int ph, int a, int t, int d, int type) {
    // bi, hi are 0-based; th in {1,2,3}; ph in {1,2}; a in {0,1}; t in {1,...,T}
    return bi
    + hi * D1
    + (th - 1) * D1 * D2 // since theta in {1,2,3}
    + (ph - 1) * D1 * D2 * D3
    + a * D1 * D2 * D3 * D4
    + (t - 1) * D1 * D2 * D3 * D4 * D5
    + d * D1 * D2 * D3 * D4 * D5 * D6 
    + type * D1 * D2 * D3 * D4 * D5 * D6 * D7; // remember d, type in {0, 1}, so no subtracting 1 needed
  };
  
  // Helper for policy index (copying structure from solve_model)
  auto idxP2 = [&](int bi, int hi, int ph, int a, int t, int d, int tau, int type) {
    // bi, hi are 0-based; th in {1,2,3}; ph in {1,2}; a in {0,1}; t in {1,...,T}
    return bi
    + hi * D21
    + (ph - 1) * D21 * D22 // since theta in {1,2,3}
    + a * D21 * D22 * D23
    + (t - 1) * D21 * D22 * D23 * D24
    + d * D21 * D22 * D23 * D24 * D25
    + tau * D21 * D22 * D23 * D24 * D25 * D26
    + type * D21 * D22 * D23 * D24 * D25 * D26 * D27; // remember d, tau, type in {0, 1}, so no subtracting 1 needed
  };
  
  // Belief grid assumptions: b_grid[i] ~ i * 0.01
  int max_b_idx = D1 - 1;
  int max_h_idx = D2 - 1;
  int Tmax = D6 - 1; // since DP uses [1..T], and dim is T+1
  if (max_age > Tmax) {
    Rcpp::stop("Simulation max_age=%d exceeds policy horizon Tmax=%d. Re-solve model with T >= max_age.",
               max_age, Tmax);
  }
  if (start_age < 1) {
    Rcpp::stop("start_age must be >= 1 because policy indexing uses (t-1).");
  }
  
  // Pre-allocate maximum possible rows: everyone lives to max_age.
  int max_T = max_age - start_age + 1;
  int max_rows = N * max_T;
  
  std::vector<int>   out_id;   out_id.reserve(max_rows);
  std::vector<int>   out_age;  out_age.reserve(max_rows);
  std::vector<double> out_b;   out_b.reserve(max_rows);
  std::vector<int>   out_h;    out_h.reserve(max_rows);
  std::vector<int>   out_theta; out_theta.reserve(max_rows);
  std::vector<int>   out_sympt; out_sympt.reserve(max_rows);
  std::vector<int>   out_phi;   out_phi.reserve(max_rows);
  std::vector<int>   out_s;     out_s.reserve(max_rows);
  std::vector<int>   out_x;     out_x.reserve(max_rows);
  std::vector<int>   out_d;     out_d.reserve(max_rows); // diagnosis state
  std::vector<int>   out_tau;     out_tau.reserve(max_rows); // years since diagnosis
  std::vector<int>   out_indiv_type;     out_indiv_type.reserve(max_rows); // constant type for exporting
  std::vector<int>   out_dead;  out_dead.reserve(max_rows);
  
  // Helper: map belief (0-1) to b-grid index
  auto belief_to_index = [&](double b) {
    if (b <= 0.0) return 0;
    if (b >= 1.0) return max_b_idx;
    int idx = (int)std::round(b * 100.0);
    if (idx < 0) idx = 0;
    if (idx > max_b_idx) idx = max_b_idx;
    return idx;
  };
  
  // Helper: map h to nearest h_grid index (assuming contiguous integer grid)
  auto h_to_index = [&](int h) {
    if (h <= h_grid[0]) return 0;
    if (h >= h_grid[max_h_idx]) return max_h_idx;
    // assuming 1-step grid; can map by offset
    int idx = h - h_grid[0];
    if (idx < 0) idx = 0;
    if (idx > max_h_idx) idx = max_h_idx;
    return idx;
  };
  
  // Main simulation loop over individuals
  for (int i = 0; i < N; ++i) {
    int id = i + 1;
    double b = b0[i];
    int h = h0[i];
    int theta = theta0[i]; // 1=no cancer, 2=early, 3=late
    int phi = phi0[i];     // 1=no chronic, 2=chronic
    int age = start_age;
    int d = 0; // start with undiagnosed cancer 
    int alive = 1;
    int tau = 0; // no years since diagnosis
    int indiv_type = type_vec[i] - 1; // initial type, fixed over individuals (convert to Rcpp indexing)
    
    // Simulate year by year until death or max_age
    while (alive && age <= max_age) {
      
      // draw symptoms based on theta -- when theta = 2, symptoms happen with probability 0.064
      double sympt_draw = R::runif(0.0, 1.0);
      int sympt = 0; // no symptoms is the default
      
      // Map continuous states to indices
      int bi = belief_to_index(b);
      int hi = h_to_index(h);
      
      // Map age to DP time index t (assume t = age, bounded by Tmax)
      int t = age;
      if (t < 1) t = 1;
      if (t > Tmax) t = Tmax;
      
      // Retrieve optimal policy (s,x) for current state
      int s;
      int x;
      if (theta == 2) { 
        s = (int)policy2[idxP2(bi, hi, phi, 0, t, d, tau, indiv_type)];
        x = (int)policy2[idxP2(bi, hi, phi, 1, t, d, tau, indiv_type)];
        sympt = (sympt_draw < 0.064) ? 1 : 0;
      } else {
        s = (int)policy[idxP(bi, hi, theta, phi, 0, t, d, indiv_type)];
        x = (int)policy[idxP(bi, hi, theta, phi, 1, t, d, indiv_type)];
        if (theta == 1) { 
          sympt = (sympt_draw < 0.033) ? 1 : 0;
        } else { 
          sympt = (sympt_draw < 0.246) ? 1 : 0;
        }
      }
      
      
      // Record current state and choices (before transitions)
      out_id.push_back(id);
      out_age.push_back(age);
      out_b.push_back(b);
      out_h.push_back(h);
      out_theta.push_back(theta);
      out_sympt.push_back(sympt);
      out_phi.push_back(phi);
      out_s.push_back(s);
      out_x.push_back(x);
      out_d.push_back(d);
      out_tau.push_back(tau);
      out_indiv_type.push_back(indiv_type);
      out_dead.push_back(0); // currently alive this period
      
      // If already *diagnosed* late-stage, beliefs effectively 1 and no screening allowed
      // (as in solve_model).
      int x_eff = x;
      if (theta == 3 && d == 1) {
        x_eff = 0; // ignore screening if late-stage
        b = 1.0;   // beliefs pinned at 1
      }
      
      // ---- Draw transitions ----
      
      // 1. Update smoking history
      int h_next = h + s;
      if (h_next > h_grid[max_h_idx]) h_next = h_grid[max_h_idx];
      
      int age_next = age + 1;
      int tau_next = tau; 
      
      // 2. Chronic condition transition
      double p_phi0, p_phi1; // Pr(phi' = 1, 2)
      phi_transition(p_phi0, p_phi1, phi, h, s, age, psi_p);
      double u_phi = R::runif(0.0, 1.0);
      int phi_next = (u_phi < p_phi1) ? 2 : 1;
      
      // 3. Cancer and beliefs
      int theta_next = theta;
      double b_next = b;
      double b2;
      if (sympt == 1) { // individual assumes that cancer may be early or late stage with equal probability 
        b2 = (0.156 * b) / (0.156 * b + 0.033 * (1-b)); // std::max(0.0, std::min( (0.156 * bt) / (0.156 * bt + 0.033 * (1-bt)), 0.99)); 
      } else { 
        b2 = ((1-0.156) * b) / ((1-0.156) * b + (1-0.033) * (1-b)); // std::max(0.0, std::min( ((1-0.156) * bt) / ((1-0.156) * bt + (1-0.033) * (1-bt)), 0.99));
      }
      
      double d_next = d; // diagnosis state next period
      
      // We need (true) transition probabilities for theta
      double p_theta_10_0, p_theta_10_1, p_theta_10_2; // theta = 1, y = 0
      double p_theta_20_y0_0, p_theta_20_y0_1, p_theta_20_y0_2; 
      double p_theta_20_y1_0, p_theta_20_y1_1, p_theta_20_y1_2;
      
      theta_transition(p_theta_10_0, p_theta_10_1, p_theta_10_2, 1, h, s, age, 0, alpha_p, zeta_p); // delta
      theta_transition2(p_theta_20_y0_0, p_theta_20_y0_1, p_theta_20_y0_2, 0, 0, alpha_p, zeta_p); // delta
      theta_transition2(p_theta_20_y1_0, p_theta_20_y1_1, p_theta_20_y1_2, 0, 1, alpha_p, zeta_p); // delta
      
      int y = 0; // no treatment yet 
      
      if (theta == 1) {
        // No cancer → maybe new incident early-stage cancer
        // Use p_theta_10
        double u_th = R::runif(0.0, 1.0);
        if (u_th < p_theta_10_0) {
          theta_next = 1; // stay no cancer
        } else {
          theta_next = 2; // move to early-stage
          tau_next = tau_next + 1; // update years since diagnosis if new cancer incidence occurs
          if (tau_next > 14) tau_next = 14;
        }
        
        // Belief evolution:
        // this is a mix of rational expectations (using p_theta_10) and Bayesian updating
        if (x_eff == 0) {
          // no screening
          double bt1_0 = b2 + (1.0 - b2) * p_theta_10_1;
          b_next = std::max(0.001, std::min(std::round(bt1_0 * 100.0) / 100.0, 0.99));
        } else {
          // screening with result z
          // draw test result z (1=positive,0=negative)
          double p_pos = (theta == 1 ? lambda : rho); // if no cancer, false positive with pr = lambda. If cancer, true positive with pr = rho
          double u_z = R::runif(0.0, 1.0);
          int z = (u_z < p_pos) ? 1 : 0;
          
          if (z == 0) {
            // negative test: Bayesian posterior then add new incidence
            double btilde = post_b_given_z0(b2, rhohat, lambdahat);
            double bt1_10 = btilde + (1.0 - btilde) * p_theta_10_1;
            b_next = std::max(0.001, std::min(std::round(bt1_10 * 100.0) / 100.0, 0.99));
          } else {
            // positive test: assume confirmatory testing so they know theta = 1
            // then next period, they have rational expectations 
            double bt1_111 = p_theta_10_1;
            b_next = std::max(0.001, std::min(std::round(bt1_111 * 100.0) / 100.0, 0.99));
          }
        }
        
      } else if (theta == 2) {
        // Early-stage cancer; treatment y depends on detection
        int z = 0;
        tau_next = tau_next + 1; // update years since diagnosis if already diagnosed or still early-stage
        if (tau_next > 14) tau_next = 14;
        
        if (x_eff == 1) {
          // screening; detect cancer with prob rho (objective)
          double u_z = R::runif(0.0, 1.0);
          z = (u_z < rho) ? 1 : 0;
          y = (z == 1 ? 1 : 0);
          d_next = (z == 1 ? 1: d); 
        }
        
        // Select (true) theta transition probs based on treatment y
        double p_theta_20_0, p_theta_20_1; // Pr(theta_next =3 omitted because colinear)
        if (y == 1) {
          p_theta_20_0 = p_theta_20_y1_0;
          p_theta_20_1 = p_theta_20_y1_1;
        } else {
          p_theta_20_0 = p_theta_20_y0_0;
          p_theta_20_1 = p_theta_20_y0_1;
        }
        
        // p_theta_20 = [Pr(theta_next=1), Pr(theta_next=2), Pr(theta_next=3)]
        double u_th = R::runif(0.0, 1.0);
         if (u_th < p_theta_20_0 + p_theta_20_1) {
          theta_next = 2;
        } else {
          theta_next = 3;
        }
        
        // Belief evolution:
        if (d_next == 1) { // if diagnosed
          b_next = 1.0; // certainty
        } else if (x_eff == 0) {
          // no screening
          double bt1_0 = b2 + (1.0 - b2) * p_theta_10_1;
          b_next = std::max(0.001, std::min(std::round(bt1_0 * 100.0) / 100.0, 0.99));
        } else {
          // screening
          if (y == 0) {
            // screened but not detected (z=0)
            double btilde = post_b_given_z0(b, rhohat, lambdahat);
            double bt1_10 = btilde + (1.0 - btilde) * p_theta_10_1;
            b_next = std::max(0.001, std::min(std::round(bt1_10 * 100.0) / 100.0, 0.99));
          } else {
            // detected; jump belief up (assume confirmatory testing occurs)
            // NumericVector p_theta_21 = theta_transition(2, h, s, age, 1, alpha, zeta); // delta
            double bt1_112 = p_theta_20_y1_1;
            b_next = std::max(0.001, std::min(std::round(bt1_112 * 100.0) / 100.0, 0.99));
          }
        }
        
      } else { // theta == 3 (late-stage)
        // Late-stage cancer; treatment y depends on detection
        int z = 0;
        tau_next = tau_next + 1; // update years since diagnosis if already diagnosed or still early-stage
        if (tau_next > 14) tau_next = 14;
        
        if (x_eff == 1) {
          // screening; detect cancer with prob rho (objective)
          double u_z = R::runif(0.0, 1.0);
          z = (u_z < rho) ? 1 : 0;
          y = (z == 1 ? 1 : 0);
          d_next = (z == 1 ? 1 : d);
        } 
            
        // No transition probs, theta = 3 forever 
        theta_next = 3; 
        
        // Belief evolution:
        if (d_next == 1) { // if diagnosed
          b_next = 1.0; // certainty
        } else if (x_eff == 0) {
          // no screening
          double bt1_0 = b2 + (1.0 - b2) * p_theta_10_1;
          b_next = std::max(0.001, std::min(std::round(bt1_0 * 100.0) / 100.0, 0.99));
        } else {
          // screening
          if (y == 0) {
            // screened but not detected (z=0)
            double btilde = post_b_given_z0(b, rhohat, lambdahat);
            double bt1_10 = btilde + (1.0 - btilde) * p_theta_10_1;
            b_next = std::max(0.001, std::min(std::round(bt1_10 * 100.0) / 100.0, 0.99));
          } else {
            // theta = 3 detected, which is absorbing;
            b_next = 1.0; 
          }
        }
      }
      
      // 4. Death
      const double survive_next = survival_prob(theta_next, phi_next, age_next, omega_p);
      double u_d = R::runif(0.0, 1.0);
      int dead_next = (u_d > survive_next) ? 1 : 0;
      
      if (dead_next == 1) {
        // record one extra row for death at age_next if desired
      //  out_id.push_back(id);
      //  out_age.push_back(age_next);
      //  out_b.push_back(b_next);
      //  out_h.push_back(h_next);
      //  out_theta.push_back(theta_next);
      //  out_phi.push_back(phi_next);
      //  out_s.push_back(s);
      //  out_x.push_back(x);
      //  out_d.push_bach(d); 
     //   out_dead.push_back(1);
        
        alive = 0;
     }
      
      // Move to next period states if still alive
      h = h_next;
      theta = theta_next;
      phi = phi_next;
      b = b_next;
      age = age_next;
      d = d_next; 
      tau = tau_next; 
    } // end the while loop for alive
  } // end loop over individuals
  
  // Build DataFrame
  return Rcpp::DataFrame::create(
    Rcpp::Named("id")    = out_id,
    Rcpp::Named("age")   = out_age,
    Rcpp::Named("b")     = out_b,
    Rcpp::Named("h")     = out_h,
    Rcpp::Named("theta") = out_theta,
    Rcpp::Named("sympt") = out_sympt,
    Rcpp::Named("phi")   = out_phi,
    Rcpp::Named("s")     = out_s,
    Rcpp::Named("x")     = out_x,
    Rcpp::Named("d")     = out_d,
    Rcpp::Named("tau")   = out_tau,
    Rcpp::Named("dead")  = out_dead,
    Rcpp::Named("type")  = out_indiv_type
  );
}