function [theta, J_history] = gradientDescentMultiReg(Xdata...
    , y, theta, alpha, num_iters,lambda)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(Xdata, y, theta, alpha, num_iters) updates 
%   theta by taking num_iters gradient steps with learning rate alpha
% Input:
%   Xdata- input data, size n×D
%   Y- target Y values for input data
%   theta- initial theta values, size D×1
%   alpha- learning rate
%   num_iters- number of iterations 
%       Where n is the number of samples, and D is the dimension 
%       of the sample plus 1 (the plus 1 accounts for the constant column)
% Output:
%   theta- the learned theta
%   J_history- The least squares cost after each iteration

% Initialize some useful values
J_history = zeros(num_iters, 1);
    
    for iter = 1:num_iters
        thetaTemp = theta;
        for i = 1: size(theta)
            if i == 1
                
            
                thetaTemp(i) = theta(i) - alpha*(sum((Xdata*theta-y).*Xdata(:,i))/(length(y)));            
            
            else
                thetaTemp(i) = theta(i) - alpha*((sum((Xdata*theta-y).*Xdata(:,i)) + lambda*thetaTemp(i))/(length(y)));            

            end
        end
        theta = thetaTemp;
        J_history(iter) = computeCostReg(Xdata, y, theta, lambda);
        
    end

end