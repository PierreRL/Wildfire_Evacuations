data {
  int<lower=1> N;
  int<lower=1> P; // number of provinces
  array[N] int<lower=1, upper=P> province;
  array[N] int<lower=0, upper=1> y;
  vector[N] log_fire_size;
  array[N] int<lower=0, upper=1> fn_indicator;
}

parameters {
  real alpha;
  vector[P] a_prov;           // varying intercepts for provinces

  real beta_log_fire_size;
  real beta_fn;
}

model {
  vector[N] theta;

  // Priors
  alpha ~ normal(0, 2);
  a_prov ~ normal(0, 1);

  beta_log_fire_size ~ normal(0, 1);
  beta_fn ~ normal(0, 1);

  // Linear predictor with province varying intercepts
  for (n in 1:N) {
    theta[n] = alpha
             + a_prov[province[n]]
             + beta_log_fire_size * log_fire_size[n]
             + beta_fn * fn_indicator[n];
  }

  // Likelihood
  y ~ bernoulli_logit(theta);
}

generated quantities {
  vector[N] log_lik;
  array[N] int<lower=0, upper=1> y_rep;

  for (n in 1:N) {
    real theta_n = alpha
                   + a_prov[province[n]]
                   + beta_log_fire_size * log_fire_size[n]
                   + beta_fn * fn_indicator[n];
    log_lik[n] = bernoulli_logit_lpmf(y[n] | theta_n);
    y_rep[n] = bernoulli_logit_rng(theta_n);
  }
}
