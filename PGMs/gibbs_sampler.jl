#Gibbs sampling for a Gaussian mixture. FOGM, 2015. October 29, 2015.
#Author: Tom Blazejewski.

#Implement Gibbs sampling for a Gaussian mixture (collapsed or uncollapsed). You can implement a
#different mixture model instead if desired. Apply code to some data, real or simulated.
#Work in log-space, use log-sum-exp trick if needed.
println("Do GIBBS")
using Distributions

function eq_23(sigma, lambda, n, z, x, k)
  #So we sample a mu_k given some info on observations and states.
  #I think we need to remove the data point we're conditioning on...?
  n_k = sum(z[:,k])
  x_bar_k = sum(x[:,k] .* x) / n_k
  mu_hat_k = ((n_k / sigma^2) / ((n_k / sigma^2) + (1/lambda^2))) * x_bar_k
  lambda_hat_k = ((n_k / sigma^2) + (1/(lambda^2)))^-1
  #mu_hat_k, lambda_hat_k
  my_gauss = Normal(mu_hat_k, lambda_hat_k)
  return rand(my_gauss)
end

function eq_21(x, mus, sigma)
  #So I somehow need to sample from the distributions given some data point.
  #Not sure how to actually code this, which is a little annoying. Maybe tomorrow will bring
  #better things.
  centers = Gaussian
end

function gen_data()
  #If there's a smarter way to do this, I don't know it unfortunately.
  sig = 0.5
  x_1 = Normal(rand(), sig)
  y_1 = Normal(rand(), sig)
  x_2 = Normal(-rand(), sig)
  y_2 = Normal(rand(), sig)
  x_3 = Normal(rand(), sig)
  y_3 = Normal(-rand(), sig)
  x_v_1 = rand(x_1, 250)
  x_v_2 = rand(x_2, 250)
  x_v_3 = rand(x_3, 250)
  y_v_1 = rand(y_1, 250)
  y_v_2 = rand(y_2, 250)
  y_v_3 = rand(y_3, 250)
  return [x_v_1 y_v_1; x_v_2 y_v_2; x_v_3 y_v_3]
end

function main()
  #Generate data.
  data = gen_data()
  rows = [1:size(data)][2]
  row_indices = rows[1:rows;]
  data = data[shuffle(row_indices), :]
  sigma = 1.0
  lambda = 0.5

  #Initiliaze mu's, where our centers start.
  x_1 = Normal(rand() * 0.01, 0.05); y_1 = Normal(rand() * 0.01, 0.05)
  x_2 = Normal(rand() * 0.01, 0.05); y_2 = Normal(rand() * 0.01, 0.05)
  x_3 = Normal(rand() * 0.01, 0.05); y_3 = Normal(rand() * 0.01, 0.05)

  mu = [x_1 y_1; x_2 y_2; x_3 y_3]

  for i in 1:rows
    eq_21()
  end
  for i in 1:K
    #sigma, lambda, n, z, x, k
    eq_23(sigma, lambda, 750, data, 3)
  end
end

println(gen_data()[1:10, :])

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
