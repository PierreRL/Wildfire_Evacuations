data {
  int<lower=1> N;
  int<lower=1> P; // provinces
  array[N] int<lower=1, upper=P> province;

  array[N] int<lower=0, upper=1> y;

  // numeric predictors
  vector[N] log_fire_size;
  vector[N] dist_to_fn_km;
  array[N] int<lower=0> n_fn_20km;
  array[N] int<lower=0, upper=1> fn_indicator;

  // categorical predictors encoded as 1..K indices
  int<lower=1> K_cause;
  array[N] int<lower=1,upper=K_cause> fire_cause;

  int<lower=1> K_type;
  array[N] int<lower=1,upper=K_type> fire_type;

  int<lower=1> K_response;
  array[N] int<lower=1,upper=K_response> response_type;

  int<lower=1> K_prot;
  array[N] int<lower=1,upper=K_prot> protection_zone;
}

parameters {
  real alpha;

  // province varying intercepts (non-centered)
  vector[P] z_prov;
  real<lower=0> sigma_prov;

  // categorical varying intercepts (non-centered)
  vector[K_cause] z_cause;
  real<lower=0> sigma_cause;

  vector[K_type] z_type;
  real<lower=0> sigma_type;

  vector[K_response] z_response;
  real<lower=0> sigma_response;

  vector[K_prot] z_prot;
  real<lower=0> sigma_prot;

  // continuous effects
  real beta_log_fire_size;
  real beta_dist;
  real beta_n_fn;

  // FN effect hierarchical by response type (non-centered)
  real beta_fn; // global mean
  vector[K_response] z_beta_fn_response;
  real<lower=0> sigma_fn_response;
}

model {
  vector[N] theta;

  // Priors
  alpha ~ normal(0, 2);

  sigma_prov ~ normal(0, 1);
  z_prov ~ normal(0, 1);

  sigma_cause ~ normal(0, 1);
  z_cause ~ normal(0, 1);

  sigma_type ~ normal(0, 1);
  z_type ~ normal(0, 1);

  sigma_response ~ normal(0, 1);
  z_response ~ normal(0, 1);

  sigma_prot ~ normal(0, 1);
  z_prot ~ normal(0, 1);

  beta_log_fire_size ~ normal(0, 1);
  beta_dist ~ normal(0, 1);
  beta_n_fn ~ normal(0, 1);

  beta_fn ~ normal(0, 1);
  sigma_fn_response ~ normal(0, 1);
  z_beta_fn_response ~ normal(0, 1);

  for (n in 1:N) {
    theta[n] = alpha
             + (sigma_prov * z_prov)[province[n]]
             + (sigma_cause * z_cause)[fire_cause[n]]
             + (sigma_type * z_type)[fire_type[n]]
             + (sigma_response * z_response)[response_type[n]]
             + (sigma_prot * z_prot)[protection_zone[n]]
             + beta_log_fire_size * log_fire_size[n]
             + beta_dist * dist_to_fn_km[n]
             + beta_n_fn * n_fn_20km[n]
             + (beta_fn + sigma_fn_response * z_beta_fn_response)[response_type[n]] * fn_indicator[n];
  }

  y ~ bernoulli_logit(theta);
}

generated quantities {
  vector[N] log_lik;
  array[N] int<lower=0, upper=1> y_rep;

  for (n in 1:N) {
    real theta_n = alpha
                   + (sigma_prov * z_prov)[province[n]]
                   + (sigma_cause * z_cause)[fire_cause[n]]
                   + (sigma_type * z_type)[fire_type[n]]
                   + (sigma_response * z_response)[response_type[n]]
                   + (sigma_prot * z_prot)[protection_zone[n]]
                   + beta_log_fire_size * log_fire_size[n]
                   + beta_dist * dist_to_fn_km[n]
                   + beta_n_fn * n_fn_20km[n]
                   + (beta_fn + sigma_fn_response * z_beta_fn_response)[response_type[n]] * fn_indicator[n];
    log_lik[n] = bernoulli_logit_lpmf(y[n] | theta_n);
    y_rep[n] = bernoulli_logit_rng(theta_n);
  }
}
