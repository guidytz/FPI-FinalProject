import numpy as np
import cv2
import functions as fn
import math

img = cv2.imread('statue_very_small.png')

sigma_s = 200
sigma_r = 2
it = 3

[alt,lar,can]  = np.array(img).shape

print(alt,lar,can)

img_norm = img/255

dIdx = np.zeros((alt,lar))
dIdy = np.zeros((alt,lar))

dIcdx = np.diff(img_norm, 1, 1)
dIcdy = np.diff(img_norm, 1, 0)

for i in range(can):
    dIdx[:,1:] = dIdx[:,1:] + abs(dIcdx[:,:,i])
    dIdy[1:,:] = dIdy[1:,:] + abs(dIcdy[:,:,i])

dHdx = (1 + sigma_s/sigma_r * dIdx)
dVdy = np.transpose(1 + sigma_s/sigma_r * dIdy)

sigma_H = sigma_s

for i in range(can):
    sigma_H_i = sigma_H * math.sqrt(3) * (2**(it-(i+1))) / math.sqrt(4**it -1)
    print(sigma_H_i)


"""     N = num_iterations;
    F = I;
    
    sigma_H = sigma_s;
    
    for i = 0:num_iterations - 1
    
        % Compute the sigma value for this iteration (Equation 14 of our paper).
        sigma_H_i = sigma_H * sqrt(3) * 2^(N - (i + 1)) / sqrt(4^N - 1);
    
        F = TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i);
        F = image_transpose(F);
    
        F = TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i);
        F = image_transpose(F);
        
    end
    
    F = cast(F, class(img));

end

%% Recursive filter.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function F = TransformedDomainRecursiveFilter_Horizontal(I, D, sigma)

    % Feedback coefficient (Appendix of our paper).
    a = exp(-sqrt(2) / sigma);
    
    F = I;
    V = a.^D;
    
    [h w num_channels] = size(I);
    
    % Left -> Right filter.
    for i = 2:w
        for c = 1:num_channels
            F(:,i,c) = F(:,i,c) + V(:,i) .* ( F(:,i - 1,c) - F(:,i,c) );
        end
    end
    
    % Right -> Left filter.
    for i = w-1:-1:1
        for c = 1:num_channels
            F(:,i,c) = F(:,i,c) + V(:,i+1) .* ( F(:,i + 1,c) - F(:,i,c) );
        end
    end

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function T = image_transpose(I)

    [h w num_channels] = size(I);
    
    T = zeros([w h num_channels], class(I));
    
    for c = 1:num_channels
        T(:,:,c) = I(:,:,c)';
    end
    
end """