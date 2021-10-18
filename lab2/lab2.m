if ~exist("results", "dir")
   mkdir("results")
end

%% 3.1 Edge Detection

img = imread("images/maccropped.jpg");
img = rgb2gray(img);
imwrite(img, "results/01_mac.png");

x_sobel = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
y_sobel = x_sobel';
gx = conv2(double(img), double(x_sobel));
gy = conv2(double(img), double(y_sobel));
imwrite(uint8(normalize_edge(gx)), "results/01_gx.png");
imwrite(uint8(normalize_edge(gy)), "results/01_gy.png");

g = sqrt(gx .^ 2 + gy .^ 2);
g = normalize_edge(g);
imwrite(uint8(g), "results/01_grad.png");

T = [10, 50, 100, 150];
for i=1:length(T)
    t = T(i);
    e = g > t;
    imwrite(e, sprintf("results/01_t_%d.png", t));
end

tl = 0.04;
th = 0.1;
sigma = 1.0;

S = [1, 2, 3, 4, 5];
for i=1:length(S)
    s = S(i);
    e = edge(img, 'canny', [tl, th], s);
    imwrite(e, sprintf("results/01_canny_sigma_%d.png", s));
end

TL = [0.01, 0.03, 0.05, 0.07, 0.09];
for i=1:length(TL)
    tl = TL(i);
    e = edge(img, 'canny', [tl, th], sigma);
    imwrite(e, sprintf("results/01_canny_tl_%d.png", round(tl * 100)));
end

%% 3.2 Hough Transform

img = imread("images/maccropped.jpg");
img = rgb2gray(img);

tl = 0.04;
th = 0.1;
sigma = 1.0;
e = edge(img, 'canny', [tl, th], sigma);
imwrite(e, "results/02_canny.png");

% unlike normal hough transform where the origin is at the top left of
% image, radon's origin is at the center of image
THETA = 0:179;
[H, xp] = radon(e); 
imagesc(THETA, xp, H);
xlabel('\theta (degrees)');
ylabel('\rho (radial distance)');
colormap("default");
colorbar;
saveas(gcf, "results/02_radon.png");

[~, idx] = max(H(:));
[radius_max, theta_max] = ind2sub(size(H), idx);
theta = THETA(theta_max)
radius = xp(radius_max)

% let (x', y') be coordinates based on origin at the center of image
% let (x, y) be image coordinates
% The current equation we could use is:
% x' cos(theta) + y' cos(theta) = radius
% Let A = radius * cos(theta), B = radius * sin(theta)
% Ax' + By' = radius ^ 2
% The above equation is based on origin at the center of iamge
% To get the equation in terms of image coordinates, 
% we have to rewrite the equation such that:
% Let X = width of image, Y = height of image
% x' = x - X/2, y' = y - Y/2
% A(x - X/2) + B(y - Y/2) = radius ^ 2
% Ax + By = radius ^ 2 + AX/2 + BY/2
% Thus, we get C = radius ^ 2 + AX/2 + BY/2

[A, B] = pol2cart(theta * pi / 180, radius);
A
B = -B  % B is negated because y-axis is pointing downwards
[Y, X] = size(img)
C = radius ^ 2 + A * X / 2 + B * Y / 2

xl = 0;
xr = X - 1;
yl = (C - A * xl) / B
yr = (C - A * xr) / B

imshow(img);
line([xl, xr], [yl, yr], 'color', [1, 0, 0]);
saveas(gcf, "results/02_superimpose.png");

%% 3.3 3D Stereo

img_l = imread('images/corridorl.jpg'); 
img_l = rgb2gray(img_l);

img_r = imread('images/corridorr.jpg');
img_r = rgb2gray(img_r);

imwrite(img_l, 'results/03_corridorl.png');
imwrite(img_r, 'results/03_corridorr.png');

D = disparity_map(img_l, img_r, 11, 11);
imagesc(D);
colormap(flipud(gray));
colorbar;
saveas(gcf, "results/03_corridor_disp.png");

img_l = imread('images/triclopsi2l.jpg'); 
img_l = rgb2gray(img_l);

img_r = imread('images/triclopsi2r.jpg');
img_r = rgb2gray(img_r);

imwrite(img_l, 'results/03_triclopsi2l.png');
imwrite(img_r, 'results/03_triclopsi2r.png');

D = disparity_map(img_l, img_r, 11, 11);
imagesc(D);
colormap(flipud(gray));
colorbar;
saveas(gcf, "results/03_triclopsi_disp.png");



%% Functions
function edges = normalize_edge(edges)
    edges = abs(edges);
    edges = edges ./ max(edges(:));
    edges = edges .* 255;
end

function disparity = disparity_map(image_l, image_r, x_temp, y_temp)
    % image_l and image_r must have the same size
    [yl, xl] = size(image_l);
    [yr, xr] = size(image_r);
    assert(xl == xr & yl == yr);

    dim_x = floor(x_temp / 2);
    dim_y = floor(y_temp / 2);
    constraint = 15;
    disparity = ones(yl, xl);

    for y=1:yl
        for x=1:xl
            yl_start = max(1, y - dim_y);
            yl_end = min(yl, y + dim_y);
            xl_start = max(1, x - dim_x);
            xl_end = min(xl, x + dim_x);
            T = image_l(yl_start:yl_end, xl_start:xl_end);
            % To calculate S(x,y), we need to perform correlation
            % correlation = convolution with inverted kernel
            T = double(rot90(T, 2));

            yr_start = yl_start;
            yr_end = yl_end;
            xr_start = max(x - dim_x - constraint, 1);
            xr_end = min(x + dim_x + constraint, xr);
            I = double(image_r(yr_start:yr_end, xr_start:xr_end));

            S = conv2(I .^ 2, ones(size(T)), 'valid') + sum(T .^ 2, 'all') - 2 * conv2(I, T, 'valid');
            [~, idx] = min(S);
            d = xr_start - xl_start + idx - 1;
            disparity(y, x) = d;
        end
    end

end
