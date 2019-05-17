function [feretDiam] = feret_diameter(S,theta)
%FERET_DIAMETER Feret diameter of a particle(s) for a given direction(s)
%
%   This function is designed to be computationally fast, but is memory
%   costly. Memory usage is dependent on the total perimeter of all labeled
%   objects in an image and the number of angles provided by theta.
%
%   feretDiam = feret_diameter(S, theta);
%   Compute the Feret diameter for objects in labeled image S at angles
%   (in degrees) given by theta. The theta input can be either a vector of
%   angles or a scalar.
%
%   The output value, feretDiam, is a cell array where each cell contains
%   the feret diameters of the corresponding objects at each of the angles
%   in theta.
%   
%   EXAMPLE
%   -------
%   % Assuming S is a labeled image...
%   feretDiam = feret_diameter(S,1:180) % calc feret diam from 1D to 180D
%   disp(feretDiam{1}) % get all feret diameters for first object
%   
%   % Find max feret diameter for first object, and the angle it occured at
%   [maxDiam, angle] = max(feretDiam{1});
%
% Author: Nicholas Schaub
% e-mail: nicholas.schaub@nist.gov
% Created: 2016-09-02, using Matlab (R2016a)

% Grab border of objects
obj_edges = boxBorderSearch(S,3);

% Get indices and label of all pixels
[y,x,l] = find(obj_edges);
clear obj_edges

% Sort pixels by label
[l, index] = sort(l);
y = y(index);
x = x(index);

% Get number of pixels for each object
l_ind = find([1; diff(l); 1]);
l_counts = l_ind(2:end) - l_ind(1:end-1);

% Create cell with x, y positions of each objects border
l_pos = mat2cell([x y],l_counts,2);

% Center points based on object centroid
l_pos = cellfun(@(xy)[xy - repmat(mean(xy,1),length(xy),1)],l_pos,'UniformOutput',false);

% Create transformation matrix
rot_trans = [cosd(theta(:)) -sind(theta(:))];

% Calculate rotation positions
l_pos = cellfun(@(xy)[rot_trans*xy'],l_pos,'UniformOutput',false);

% Get Ferets diameter (add 1 pixel to account for pixel width)
feretDiam = cellfun(@(xy)[max(xy,[],2)-min(xy,[],2)+sum(abs(rot_trans),2)],l_pos,'UniformOutput',false);

end