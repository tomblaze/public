#Gibbs sampling for a Gaussian mixture. FOGM, 2015. October 29, 2015.
#Author: Tom Blazejewski.

#Implement Gibbs sampling for a Gaussian mixture (collapsed or uncollapsed). You can implement a
#different mixture model instead if desired. Apply code to some data, real or simulated.
#Work in log-space, use log-sum-exp trick if needed.
println("Do GIBBS")
using Distributions
using PDMats
using PyPlot

function gen_data()
  #Generate three multivariate
  sig = 1.0
  dims = 2 #I don't imagine this'll ever change...
  u_1 = randn(dims) * 3
  u_2 = randn(dims) * 3
  u_3 = randn(dims) * 3
  center_1 = MvNormal(u_1, sig)
  center_2 = MvNormal(u_2, sig)
  center_3 = MvNormal(u_3, sig)
  #Sampling from MvNormal gives array like dims * n... that seems unnatural, right?
  #Why not do "each row is datapoint"? Going to transpose.
  samples = [rand(center_1, 1000) rand(center_2, 1000) rand(center_3, 1000)]'
  println("Did generate...")
  println(size(samples))
  println(center_1)
  println(center_2)
  println(center_3)
  return u_1, u_2, u_3, shuffle_rows(samples) #want to randomize order.
end

function color_generator(n)
  r = (255 * n) / 100
  g = (255 * (100 - n)) / 100
  b = 0
  return r, g, b
end

function categorical_to_one_hot(z)
  top_cat = maximum(z) #maximum category but let's assume we start at 1.
  big_array = zeros(Int64, length(z), int(top_cat))
  for val in 1:length(z)
    big_array[val, z[val]] = 1
  end
  return big_array
end

function eq_23(sigma, lambda, n, z, x, k)
  z = categorical_to_one_hot(z)
  n_k = sum(z[:,k])

  x_bar_k = sum(z[:,k] .* x, 1) / n_k
  mu_hat_k = ((n_k / sigma^2) / ((n_k / sigma^2) + (1/lambda^2))) .* x_bar_k
  lambda_hat_k = ((n_k / sigma^2) + (1/(lambda^2)))^-1

  my_gauss = MvNormal(vec(mu_hat_k), lambda_hat_k)

  return rand(my_gauss)
end

function log_sum_exp(vec)
  the_max = maximum(vec)
  s = 0.0
  for i in 1:length(vec)
    s += exp(vec[i] - the_max)
  end
  return log(s) + the_max
end

function total_prob(x, z_mat, mus, sigma, K)
  centers = MvNormal[]
  for i in 1:K
    push!(centers, MvNormal(vec(mus[i, :]), sigma))
  end
  big_prob = 0
  for i in 1:size(x)[1]
    #this is the part that remains incorrect.
    #need to consider the probability of the z, sum these...
    center_probs = [log(1/3) + z_mat[i, j] + logpdf(centers[j], vec(x[i, :])) for j in 1:K]
    big_prob += log_sum_exp(center_probs)
  end
  return big_prob
end

function eq_21(x, mus, sigma, K)
  #So I somehow need to sample from the distributions given some data point.
  #Not sure how to actually code this, which is a little annoying. Maybe tomorrow will bring
  #better things.
  #Summing the itsy bitsy probabilities here is where the log-sum-exp trick comes in to play.
  centers = MvNormal[]
  #println("mus: ", mus)
  #mus = mus'
  for i in 1:K
    #println(mus[i, :])
    push!(centers, MvNormal(vec(mus[i, :]), sigma))
  end
  z = zeros(size(x)[1])
  #println("size(x): ", size(x)[1])
  z_mat = zeros(size(x)[1], K)
  for i in 1:size(x)[1]
    #for each point in x...
    #println("x ", x[i, :])
    #println(logpdf(centers[1], vec(x[i, :])))
    #I need to acquire this, center_probs, for downtream probability measurement.
    center_probs = [logpdf(centers[j], vec(x[i, :])) for j in 1:K]
    partition = log_sum_exp(center_probs)
    #println([exp(a - partition) for a in center_probs])
    prob_vec = float([exp(a - partition) for a in center_probs])
    z_mat[i, :] = [(a - partition) for a in center_probs]
    to_sample = Categorical(prob_vec)
    z[i] = rand(to_sample)
  end
  return z_mat, z
end

function shuffle_rows(data)
  rows = size(data)[1]
  row_indices = [1:rows;]
  data = data[shuffle(row_indices), :]
  return data
end

function main()
  #Generate data.
  rand_int = int(rand() * 10000)
  #println("rand int:", rand_int)
  #
  srand(rand_int)
  #srand(761)
  #srand(9561)
  #srand(6698)
  #srand(4863)
  #srand(5110) #genuinely doesn't quite get there...
  #srand(4734) #oh my god, the chosen one!
  #8196 pretty interesting
  #srand(8196)
  #879 is a good general one.
  u_1, u_2, u_3, data = gen_data()
  println("data: ", size(data))
  scatter(data[:,1], data[:,2])
  println("Starting")
  sigma = 1.0
  lambda = 0.5
  K = 3
  #Initiliaze mu's, where our centers start.
  mu = rand(MvNormal(randn(2), 2), K)'

  scatter(mu[:,1], mu[:,2], c="yellow", s=200)
  scatter(u_1[1], u_1[2], c="white", s=300)
  scatter(u_2[1], u_2[2], c="white", s=300)
  scatter(u_3[1], u_3[2], c="white", s=300)

  all_probs = Float64[]

  num_iter = 50
  for count in 1:num_iter #100 iterations is just a place-holder for a more reasonable number of iterations in future.
    if count % 1 == 0
      println(count)
      r, g, b=color_generator((count/num_iter) * 100)
      scatter(mu[:,1], mu[:,2], c=@sprintf("#%02X%02X%02X", r, g, b), s=50)
    end
    z_mat, z = eq_21(data, mu, sigma, K)
    push!(all_probs, total_prob(data, z_mat, mu, sigma, K))
    mus = zeros(K, 2)
    for i in 1:K
      mu = eq_23(sigma, lambda, 3000, z, data, i)
      mus[i, :] = mu
    end
    mu = mus
  end
  println(mu)
  println("rand_int: ", rand_int)
  close()
  plot(1:num_iter, all_probs)
end

main()

