% Author: Noemi Rieben, Kunxin Wu 
% Date: 7th May 2025
% Description: A code to optimize a robotic basketball shooter 

    % Unified Genetic Algorithm-based optimizer (no toolbox required)
    close all; clc;

    %% Parameters
    g = 9.81; % m/s^2
    t_delta_max = 2; % soft boundary how long the motion should take
    max_angle_w1 = 90;
    max_angle_w2 = -90; % degrees, max angular impulses w1 and w2
    target_s = 0.25; % m, target displacement at the end of the motion
    target_v = 30; % m/s, target velocity at the end of the motion
    target_alpha = -30; % degrees, target angle at the end of the motion
    target = [target_v, target_alpha, target_s];
    nvars = 6; % number of variables to optimize
    lb = [0, 0, 0, 0, 0, max_angle_w2];
    ub = [t_delta_max, t_delta_max, t_delta_max, t_delta_max, max_angle_w1, 0];% lower and upper bounds for the 6 decision variables
    
    % settings for the genetic algorithm
    pop_size = 10; % population size
    generations = 5000;
    stop_threshold = 1e-2; % stops if error < threshold
    stagnation_limit = 3; % stops if no improvement for X generations

    %% Objective Function
    % X is [delta_t1, delta_t2, delta_t3, delta_t4, w1, w2]
    % target = [target_v0x, target_vy0, target_s]
    obj_fun = @(X) simulate_cost(X, t_delta_max, g, target);

    %% Run Genetic Algorithm
    % bestX: best solution of the 6 decision varible that gives the lowest error
    % bestF: corresponding error from obj_fun
    [bestX, bestF] = GA_optimizer(obj_fun, nvars, lb, ub, pop_size, generations, stop_threshold, stagnation_limit);

    fprintf('\nBest solution: t1=%.3f, t2=%.3f, t3=%.3f, t4=%.3f | Error = %.6f\n', bestX, bestF);

    %% Simulate and Plot Best Solution
    simulate_motion(bestX, 1);


function [bestX, bestF] = GA_optimizer(obj_fun, nvars, lb, ub, pop_size, generations, stop_threshold, stagnation_limit)
    % Genetic Algorithm implementation (no toolbox)

    error_history = zeros(1, generations);
    pop = rand(pop_size, nvars) .* (ub - lb) + lb;
    fitness = zeros(pop_size, 1);

    for g = 1:generations
        for i = 1:pop_size
            X = pop(i, :);
            fitness(i) = obj_fun(X);

            
        end
        best_error = inf;
        no_improvement_counter = 0;
        
        % sort and find the elite parent for the next generation
        [~, idx] = sort(fitness);
        pop = pop(idx, :);
        fitness = fitness(idx);
        elite = pop(1:5, :);

        % put the best error of this generation to the error history
        error_history(g) = fitness(1);
        %fprintf('GA Generation %d: Best Error = %.6f\n', g, fitness(1));

        % Check stop conditions
        if fitness(1) < best_error - 1e-6
            best_error = fitness(1);
            no_improvement_counter = 0;
        else
            no_improvement_counter = no_improvement_counter + 1;
        end

        if fitness(1) < stop_threshold
            fprintf('Stopping early at generation %d: error < threshold (%.2e)\n', g, stop_threshold);
            break;
        end

        if no_improvement_counter >= stagnation_limit
            fprintf('Stopping due to stagnation at generation %d (no improvement for %d generations)\n', g, stagnation_limit);
            break;
        end

        % generate a new population by crossover and mutation
        new_pop = elite;
        while size(new_pop, 1) < pop_size
            parents = pop(randi(10, 2, 1), :);
            alpha = rand;
            child = alpha * parents(1,:) + (1 - alpha) * parents(2,:);
            child = child + 0.1 * randn(size(child));
            child = min(max(child, lb), ub);
            new_pop(end+1, :) = child;
        end

        pop = new_pop;
        %error_history(g) = fitness(1);
        %fprintf('GA Generation %d: Best Error = %.6f\n', g, fitness(1));
    end

    % the best solution is in the newest generation
    bestX = pop(1, :);
    bestF = fitness(1);

    % Plot error history
    figure;
    plot(1:generations, error_history, 'LineWidth', 2);
    xlabel('Generation'); ylabel('Error'); title('Genetic Algorithm Error Over Generations'); grid on;
    
    MAX_Error = max(error_history)
    MIN_Error = min(error_history)
    % ylim([MIN_Error,MAX_Error]);
end

% brief: function to simulate error
% output: error of 
% input 1: the arrays of decision variables that genetic algorithm will optimize
% input 2: the maximum allowed duration for each time segment
% input 3: standard gravity
% input 4: target velocity, rotation angle, displacement at the end of
% motion
function error = simulate_cost(X, t_total, g, target)
    % extract the 6 decision variables
    delta_t1 = X(1);
    delta_t2 = X(2);
    delta_t3 = X(3);
    delta_t4 = X(4);
    w1 = X(5);
    w2 = X(6);

    % extract the targets 
    target_velocity = target(1);
    target_alpha = target(2);
    target_s = target(3);

    % use the input 6 decision variables to simulate motion
    % alpha: rad, angle over time
    % a: m/s^2, acceleration over time
    % v: m/s, velocity over time 
    % s: m, displacement over time
    % omega1: deg/s, angular velocity applied during first segment of motion
    % omega2: deg/s, angular velocity applied during the third segment of motion
    [alpha, a, v, s, omega1_deg, omega2_deg] = simulate_motion(X, 0);

    % return huge error if 
    % 1. did not reach minimum displacement
    % 2. angular velocity exceed 25 deg/s
    if s(end) <= target_s || abs(omega2_deg) > 25 || abs(omega1_deg) > 25
        error = 1e6; 
        return;
    end
    
    % 
    dt = [delta_t1, delta_t2, delta_t3, delta_t4];
    
    % span is 
    alpha_span = max(rad2deg(alpha)) - min(rad2deg(alpha));
    % error to punish if rotation is not far enough
    e_span = max(0, 130 - alpha_span);
    %e_s = abs(s(end) - target_s);
    e_time = std(dt); % the times should be not spread widely 
    % error to punish if target velocity is not achieved
    e_end_velocity = abs(target_velocity - v(end) + 1);
    % error to punish if target angle is not achieved
    e_end_angle = abs(alpha(end) - target_alpha + 1);
    
    % error that combines all of the above
    error = (10 * e_span + abs(w1) + abs(w2))+ 1000*e_end_velocity + 10000*e_end_angle;

end

% brief: use the 6 decision variable to simulate the robotic arm motion
function [alpha, a, v, s, omega1_deg, omega2_deg] = simulate_motion(X, best_solution)
    n = 2000;
    t = linspace(0, sum(X(1:4)), n);
    delta_t1 = X(1);
    delta_t2 = X(2);
    delta_t3 = X(3);
    delta_t4 = X(4);
    w1 = X(5);
    w2 = X(6);    
    t1 = delta_t1; t2 = delta_t1 + delta_t2; t3 = delta_t1 + delta_t2 + delta_t3; t4 = delta_t1 + delta_t2 + delta_t3 + delta_t4; 
    omega1_deg = w1 / delta_t1;
    omega2_deg = w2 / delta_t3;

    omega_deg = zeros(size(t));
    omega_deg(t <= t1) = omega1_deg;
    omega_deg(t > t1 & t <= t2) = 0;
    omega_deg(t > t2 & t <= t3) = omega2_deg;
    omega_deg(t > t3 & t <= t4) = 0;
    omega_deg(t > t4) = 0;

    omega = deg2rad(omega_deg);
    alpha = cumtrapz(t, omega);
    a = 9.81 * sin(alpha);
    v = cumtrapz(t, a);
    s = cumtrapz(t, v);
    
    if best_solution
        % Only plot for best solution
        figure;
        subplot(4,1,1); plot(t, s, 'b', 'LineWidth', 2); ylabel('s(t) [m]'); grid on;
        title(sprintf('Displacement (%.1f cm), \\omega_1=%.2f, \\omega_2=%.2f', s(end), omega1_deg, omega2_deg));
        subplot(4,1,2); plot(t, v, 'g', 'LineWidth', 2); ylabel('v(t) [m/s]'); grid on;
        subplot(4,1,3); plot(t, rad2deg(alpha), 'r', 'LineWidth', 2); ylabel('\alpha(t) [deg]'); grid on;
        subplot(4,1,4); plot(t, omega_deg, 'm', 'LineWidth', 2); xlabel('t [s]'); ylabel('\omega(t) [deg/s]'); grid on;
    end
end
