// [[Rcpp::plugins(openmp)]]

#include <Rcpp.h>
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Returns an n x Q matrix of Q individual-level moments.

// [[Rcpp::export]]
NumericMatrix moment_diffs_cpp(
    const NumericVector& age,
    const NumericVector& s,
    const NumericVector& h,
    const NumericVector& x,
    const NumericVector& dead,
    const NumericVector& theta, 
    const NumericVector& type
) {
  const std::size_t n = age.size();
  //if (h.size() != n || s.size() != n || x.size() != n || dead.size() != n || theta.size() != n || type.size() != n) {
  //  throw std::runtime_error("compute_individual_moments_with_resid: input vectors must have equal length.");
  //}
  
  // Column-major storage: element (i,j) is outp[i + n*j].
  NumericMatrix out(n, 40);
  const double* age_p   = age.begin();
  const double* s_p     = s.begin();
  const double* h_p     = h.begin();
  const double* x_p     = x.begin();
  const double* dead_p  = dead.begin();
  const double* theta_p = theta.begin();
  const double* type_p  = type.begin();
  double* outp = out.begin();
  
  // --- 1. COUNT TYPES TO HANDLE UNEQUAL REPRESENTATION ---
  double n1 = 0, n2 = 0, n3 = 0, n4 = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (dead_p[i] == 0.0 && theta_p[i] < 3.0) {
      if      (type_p[i] == 1.0) n1++;
      else if (type_p[i] == 2.0) n2++;
      else if (type_p[i] == 3.0) n3++;
      else if (type_p[i] == 4.0) n4++;
    }
  }
  
  // Calculate Weights (N / Nk). Use 0 if group is empty to avoid division by zero.
  double w1 = (n1 > 0) ? (double)n / n1 : 0.0;
  double w2 = (n2 > 0) ? (double)n / n2 : 0.0;
  double w3 = (n3 > 0) ? (double)n / n3 : 0.0;
  double w4 = (n4 > 0) ? (double)n / n4 : 0.0;
  
  #pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    const double age_i   = age_p[i];
    const double h_i     = h_p[i];
    const double s_i     = s_p[i];
    const double x_i     = x_p[i];
    const double dead_i  = dead_p[i];
    const double theta_i = theta_p[i];
    const double type_i  = type_p[i];
    const double test_i  = (theta_i < 2.0) ? 0.0 : 1.0; // indicator for cancer (early or late)
    
    const double age2 = age_i * age_i;
    const double h2   = h_i * h_i;
    
    // helper macro for writing: OUT(col) = value;
    #define OUT(j) outp[i + n * static_cast<std::size_t>(j)]
    
    double resid;
    
    // smoking on age prevalence residual (checked 1/30/2026)
    resid = s_i - (-0.97 + 0.08 * age_i - 0.00081 * age2);  // (dead_i == 0) ? s_i - (0.97 + 0.08 * age_i - 0.00081 * age2) : 0;
    OUT(0) = (resid);           // mom0_0
    OUT(1) = (resid * age_i);   // mom0_1
    OUT(2) = (resid * age2); 
    
    // mean smoking on age (binned)
    double s_target;
    if      (age_i < 45) s_target = 0.340;
    else if (age_i < 50) s_target = 0.3382;
    else if (age_i < 55) s_target = 0.3129;
    else if (age_i < 60) s_target = 0.3334;
    else if (age_i < 65) s_target = 0.3283;
    else if (age_i < 70) s_target = 0.2583;
    else if (age_i < 75) s_target = 0.1897;
    else if (age_i < 80) s_target = 0.1399;
    else                 s_target = 0.0724;
    OUT(3) = s_i - s_target;
    
    // NOTE: these are recreated based on the JAMA moments below 
    // screening on age prevalence residual (checked 1/30/2026)
    //resid = x_i - (-1.294 + 0.0428 * age_i - 0.00027 * age2); // (dead_i == 0 && theta_i < 3) ? x_i - (-1.294 + 0.0428 * age_i - 0.00027 * age2) : 0.0;
    //out(i,4) = (resid);           // mom1_0
    //out(i,5) = (resid * age_i);   // mom1_1
    //out(i,6) = (resid * age2);    // mom1_2
    
    // mean screening on age (binned)
    //out(i,7) = (age_i < 45) ? x_i - 0.062 : 0.0; 
    //out(i,7) = (age_i >= 45 && age_i < 50) ? x_i - 0.0872 : out(i,7);
    //out(i,7) = (age_i >= 50 && age_i < 55) ? x_i - 0.1360 : out(i,7);
    //out(i,7) = (age_i >= 55 && age_i < 60) ? x_i - 0.2320 : out(i,7);
    //out(i,7) = (age_i >= 60 && age_i < 65) ? x_i - 0.3372 : out(i,7);
    //out(i,7) = (age_i >= 65 && age_i < 70) ? x_i - 0.3910 : out(i,7);
    //out(i,7) = (age_i >= 70 && age_i < 75) ? x_i - 0.3934 : out(i,7);
    //out(i,7) = (age_i >= 75 && age_i < 80) ? x_i - 0.3634 : out(i,7);
    //out(i,7) = (age_i >= 80) ? x_i - 0.2840 : out(i,7);
    
    // smoking on history prevalence residual (checked 1/30/2026)
    resid = s_i * 100.0 - (7.2068 + 0.727 * h_i - 4.83e-8 * h2);
    OUT(4) = resid;
    OUT(5) = resid * h_i;
    OUT(6) = resid * h2;
    
    // screening on history prevalence residual (checked 1/30/2026)
    resid = x_i * 100.0 - (1.783 + 0.2398 * h_i - 8.981e-8 * h2);
    OUT(7) = resid;
    OUT(8) = resid * h_i;
    OUT(9) = resid * h2;
    
    // TODO: add stigma effects here to both groups
    // FOR INELIGIBLE FOLKS: average (intent to) screening (where relevant): 
    // baseline intent to screen mean in Truveta is 0.3843
    // if you face the price, then intent to screen goes down by 0.1741 to 0.2102
    // FOR USPSTF ELIGIBLE FOLKS: average intent to screen should be 0.6457; put these in the same moment
    const bool alive_no_late = (dead_i == 0.0 && theta_i < 3.0);
    if (alive_no_late) {
      if (age_i < 50.0 || h_i < 20.0) OUT(10) = x_i - 0.2102;
      else                           OUT(10) = x_i - 0.6457;
    } else {
      OUT(10) = 0.0;
    }
    
    // similarly for stigma and optimism: Prolific suggests that effect of reducing stigma should be 0.080 points, 
    // and that effect of improving optimism should be 0.21 points. 
    // We have 4 moments/comparisons to pin down 3 parameters (including interaction term)
    
    if (alive_no_late) {
      // 1. type = 1 - type = 2 (Stigma Effect)
      if      (type_i == 1.0) OUT(11) = w1 * (x_i - 0.08); 
      else if (type_i == 2.0) OUT(11) = -w2 * x_i;
      else                    OUT(11) = 0.0;
      
      // 2. type = 3 - type = 4 (Stigma Effect)
      if      (type_i == 3.0) OUT(12) = w3 * (x_i - 0.08);
      else if (type_i == 4.0) OUT(12) = -w4 * x_i;
      else                    OUT(12) = 0.0;
      
      // 3. type = 3 - type = 1 (Optimism Effect)
      if      (type_i == 3.0) OUT(13) = w3 * (x_i - 0.21);
      else if (type_i == 1.0) OUT(13) = -w1 * x_i;
      else                    OUT(13) = 0.0;
      
      // 4. type = 4 - type = 2 (Optimism Effect)
      if      (type_i == 4.0) OUT(14) = w4 * (x_i - 0.21);
      else if (type_i == 2.0) OUT(14) = -w2 * x_i;
      else                    OUT(14) = 0.0;
    } else {
      OUT(11) = 0.0; OUT(12) = 0.0; OUT(13) = 0.0; OUT(14) = 0.0;
    }
    
    // Moments from the JAMA Network Open paper 
    // # there are 4: screening by age for low pack-year, screening by age for high pack-year, screening by pack-year for high age, and screening by pack-year for low age
    // for each of these, run linear regressions (quadratic?) above and below cutoff for eligibility 
    
    /* NumericVector target_mom7a = NumericVector::create(
      0.02, 0.0175, 0.02, 0.025, 0.03, 0.025, 0.02, 0.03, 0.04,
      0.06, 0.06, 0.035, 0.03, 0.04, 0.035, 0.08, 0.04, 0.04,
      0.06, 0.06, 0.06, 0.07, 0.07, 0.05,
      0.07, 0.07,
      0.075, 0.075, 0.075, 0.075, 0.075,
      0.04, 0.05, 0.06, 0.08, 0.075, 0.10, 0.04, 0.09, 0.11
    ); // coefficients are (b0, b1) = (-0.1189, 0.0033) for low-pack group below cutoff, and (b0, b1) = (-0.024, 0.0014) for low-pack group above cutoff
    
    NumericVector target_mom7b = NumericVector::create(
      0.04, 0.02, 0.03, 0.03, 0.02,
      0.06, 0.035, 0.035, 0.02, 0.03, 0.075, 0.075, 0.06, 0.07, 0.055, 0.1, 0.1, 0.14, 0.12,
      0.16, 0.16, 0.195, 0.18, 0.16, 0.23, 0.2, 0.25, 0.22, 0.245, 0.245, 0.25, 0.24, 0.285,
      0.28, 0.22, 0.29, 0.25, 0.36, 0.29, 0.235
    ); // coefficients are (b0, b1) = (0.03, 0) for high-pack group below cutoff, and (b0, b1) = (-0.363, 0.009) for high-pack group above cutoff
    
    NumericVector target_mom7c = NumericVector::create(
      0.01, 0.015, 0.02, 0.02, 0.02, 0.01, 0.05, 0.01, 0.02, 0.02, 0.02, 0.015, 0.015, 0.015,
      0.035, 0.04, 0.02, 0.02, 0.04, 0.03, 0.05, 0.02, 0.02, 0.02, 0.035, 0.035,
      0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.045, 0.05, 0.03, 0.06, 0.05, 0.03, 0.035, 0.04
    ); // coefficients are (b0, b1) = (0.014, 0.0008) for young group below cutoff, and (b0, b1) = (0.0083, 0.0008) for young group above cutoff
    
    NumericVector target_mom7d = NumericVector::create(
      0.02, 0.06, 0.16, 0.045, 0.06, 0.06, 0.08, 0.09, 0.07, 0.14, 0.09, 0.1,
      0.115, 0.16, 0.085, 0.075, 0.09, 0.085, 0.115, 0.12, 0.165, 0.14, 0.125,
      0.18, 0.12, 0.165, 0.155, 0.155, 0.125, 0.13, 0.14, 0.16, 0.1, 0.1, 0.13,
      0.14, 0.1, 0.145, 0.1, 0.125
    ); // coefficients are (b0, b1) = (0.065, 0.0025) for old group below cutoff, and (b0, b1) = (0.1834, -0.002) for old group above cutoff
     */
    
    // regressions for screening by age, stratified by smoking history (low-pack group) (checked 1/30/2026)
    resid = (alive_no_late && h_i < 20.0 && age_i < 50.0) ? x_i - (1.30 - 0.0607 * age_i + 0.0007 * age2) : 0.0;
    OUT(15) = resid;
    OUT(16) = resid * age_i;
    OUT(17) = resid * age2;
    resid = (alive_no_late && h_i < 20.0 && age_i >= 50.0) ? x_i - (-0.046 + 0.0020 * age_i) : 0.0;
    OUT(18) = resid;
    OUT(19) = resid * age_i;
    OUT(20) = resid * age2;
    
    // regressions for screening by age, stratified by smoking history (high-pack group)
    resid = (alive_no_late && h_i >= 20.0 && age_i < 50.0) ? x_i - (-0.5629 + 0.0269 * age_i - 0.0003 * age2) : 0.0;
    OUT(21) = resid;
    OUT(22) = resid * age_i;
    OUT(23) = resid * age2;
    resid = (alive_no_late && h_i >= 20.0 && age_i >= 50.0) ? x_i - (-1.1959 + 0.0349 * age_i - 0.0002 * age2) : 0.0;
    OUT(24) = resid;
    OUT(25) = resid * age_i;
    OUT(26) = resid * age2;
    
    // regressions for screening by smoking history, stratified by age (young group) (checked 1/30/2026)
    resid = (alive_no_late && h_i < 20.0 && age_i < 50.0) ? x_i * 100.0 - (1.667 + 0.0076 * h_i + 0.0034 * h2) : 0.0;
    OUT(27) = resid;
    OUT(28) = resid * h_i;
    OUT(29) = resid * h2;
    resid = (alive_no_late && h_i >= 20.0 && age_i < 50.0) ? x_i * 100.0 - (10.8332 - 0.6164 * h_i + 0.0116 * h2) : 0.0;
    OUT(30) = resid;
    OUT(31) = resid * h_i;
    OUT(32) = resid * h2;
    
    // regressions for screening by smoking history, stratified by age (old group)
    resid = (alive_no_late && h_i < 20.0 && age_i >= 50.0) ? x_i * 100.0 - (4.1971 + 0.8962 * h_i - 0.0324 * h2) : 0.0;
    OUT(33) = resid;
    OUT(34) = resid * h_i;
    OUT(35) = resid * h2;
    resid = (alive_no_late && h_i >= 20.0 && age_i >= 50.0) ? x_i * 100.0 - (9.332 + 0.4622 * h_i - 0.0104 * h2) : 0.0;
    OUT(36) = resid;
    OUT(37) = resid * h_i;
    OUT(38) = resid * h2;
    
    // average cancer incidence rate by age (binned);
    double test_target;
    if      (age_i < 45) test_target = 0.00165;
    else if (age_i < 50) test_target = 0.00525;
    else if (age_i < 55) test_target = 0.0105;
    else if (age_i < 60) test_target = 0.0177;
    else if (age_i < 65) test_target = 0.0358;
    else if (age_i < 70) test_target = 0.0428;
    else if (age_i < 75) test_target = 0.0644;
    else if (age_i < 80) test_target = 0.0923;
    else                 test_target = 0.112;
    OUT(39) = test_i - test_target;
  }
  
  return out;
}