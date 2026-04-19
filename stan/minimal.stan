data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> y;
  vector[N] log_fire_size;
  array[N] int<lower=0, upper=1> fn_indicator;
}

parameters {
  real alpha;
  real beta_log_fire_size;
  real beta_fn;
}

model {
  vector[N] theta;

  // Priors
  alpha ~ normal(0, 2);
  beta_log_fire_size ~ normal(0, 1);
  beta_fn ~ normal(0, 1);

  // Linear predictor
  for (n in 1:N) {
    theta[n] = alpha
             + beta_log_fire_size * log_fire_size[n]
             + beta_fn * fn_indicator[n];
  }

  // Likelihood
  y ~ bernoulli_logit(theta);
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;

  for (n in 1:N) {
    real theta_n = alpha
                   + beta_log_fire_size * log_fire_size[n]
                   + beta_fn * fn_indicator[n];
    log_lik[n] = bernoulli_logit_lpmf(y[n] | theta_n);
    y_rep[n] = bernoulli_logit_rng(theta_n);
  }
}