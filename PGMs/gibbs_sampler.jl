#Gibbs sampling for a Gaussian mixture. FOGM, 2015. October 29, 2015.
#Author: Tom Blazejewski.

#Implement Gibbs sampling for a Gaussian mixture (collapsed or uncollapsed). You can implement a
#different mixture model instead if desired. Apply code to some data, real or simulated.
#Work in log-space, use log-sum-exp trick if needed.
println("Do GIBBS")
using Distributions
using PDMats

function eq_23(sigma, lambda, n, z, x, k)
  #So we sample a mu_k given some info on observations and states.
  #I think we need to remove the data point we're conditioning on...?
  n_k = sum(z[:,k])
  x_bar_k = sum(x[:,k] .* x) / n_k
  mu_hat_k = ((n_k / sigma^2) / ((n_k / sigma^2) + (1/lambda^2))) * x_bar_k
  lambda_hat_k = ((n_k / sigma^2) + (1/(lambda^2)))^-1
  #mu_hat_k, lambda_hat_k
  my_gauss = MvNormal(mu_hat_k, lambda_hat_k)
  return rand(my_gauss)
end

function log_sum_exp(vec)
  #log_vec is a vector of logarithms (not raw probabilities) to sum.
  the_max = maximum(vec)
  s = 0.0
  for i in 1:length(vec)
    s += exp(x[i] - the_max)
  end
  return log(s) + the_max
end

function eq_21(x, mus, sigma, K)
  #So I somehow need to sample from the distributions given some data point.
  #Not sure how to actually code this, which is a little annoying. Maybe tomorrow will bring
  #better things.
  #Summing the itsy bitsy probabilities here is where the log-sum-exp trick comes in to play.
  centers = MvNormal[]
  mus = mus'
  for i in 1:K
    println(mus[i, :])
    push!(centers, MvNormal(vec(mus[i, :]), sigma))
  end
  z = zeros(size(x)[2])
  for i in 1:size(x)[2]
    #for each point in x...
    println("x ", x[i, :])
    println(logpdf(centers[1], x[i, :]))
    center_probs = [logpdf(centers[j], x[i, :]) for j in centers]
    partition = log_sum_exp(center_probs)
    to_sample = Categorical([exp(a - partition) for a in center_probs])
    z[i] = rand(to_sample)
  end
  return z
end

function shuffle_rows(data)
  println("Hello?")
  rows = size(data)[2]
  row_indices = [1:rows;]
  data = data[shuffle(row_indices), :]
  println("done")
  return data
end

function gen_data()
  #Generate three multivariate
  sig = 1.0
  dims = 2 #I don't imagine this'll ever change...
  center_1 = MvNormal(randn(dims), sig)
  center_2 = MvNormal(randn(dims), sig)
  center_3 = MvNormal(randn(dims), sig)
  #Sampling from MvNormal gives array like dims * n... that seems unnatural, right?
  #Why not do "each row is datapoint"? Going to transpose.
  samples = [rand(center_1, 250) rand(center_2, 250) rand(center_3, 250)]'
  println("Did generate...")
  return shuffle_rows(samples) #want to randomize order.
end

function main()
  #Generate data.
  data = gen_data()
  println("Starting")
  sigma = 1.0
  lambda = 0.5
  K = 3
  #Initiliaze mu's, where our centers start.
  mu = rand(MvNormal(randn(2) * 0.01, 0.05), K)
  println("Past mu")

  for count in 1:100 #100 iterations is just a place-holder for a more reasonable number of iterations in future.
    println(count)
  #for i in 1:rows
    #eq_21 already goes for all rows...
    z = eq_21(data, mu, sigma, K)
  #end
    for i in 1:K
      #sigma, lambda, n, z, x, k
      mu = eq_23(sigma, lambda, 750, z, data, K)
    end
  end
end

#println(gen_data()[1:10, :])
main()

#Typing out pseudo-code...
#Input: Data x and a number of components, K.
#Initialize: mixture locations mu.

#Main mixture locations u and mixture assignments z.

#Repeat:
  #For each i in (1, n):
    #Sample z_i | {u,z_{not_i}, x} from eq. 21
  #For each k in (1, K):
    #Sample u_k | {u_{not_k}, z, x} from eq. 23

#Equation 21: p(z_i|u, x_i) = p(z_i)p(x_i|u_{z_i}) = (pi_z) (phi(x_i', u_z, sigma^2))
#We assume pi_z = 1/K for now.

#Equation 23... u_k | z, x ~ N(u_hat_k, lambda_hat_k)
#u_hat_k = x_bar_k * ((n_k / sigma^2) / (n_k/sigma^2 + 1/lambda^2))
#
