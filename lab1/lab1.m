%% 2.1 Contrast Stretching

img = imread("images/mrttrainbland.jpg");
whos img
gray = rgb2gray(img);
min(gray(:))
max(gray(:))

% refer to appendix for function implementation
new_gray = contrast_stretching(gray);
min(new_gray(:))
max(new_gray(:))
figure("Name", "original");
imshow(gray);
figure("Name", "contrast stretching");
imshow(new_gray);

%% 2.2 Histogram Equalization 

img = imread("images/mrttrainbland.jpg");
gray = rgb2gray(img);

figure("Name", "original");
imshow(gray);
figure("Name", "original-hist10");
imhist(gray, 10);
figure("Name", "original-hist256");
imhist(gray, 256);

new_gray = histeq(gray, 255);
figure("Name", "histeq");
imshow(new_gray);
figure("Name", "histeq-hist10");
imhist(new_gray, 10);
figure("Name", "histeq-hist256");
imhist(new_gray, 256);

new_gray2 = histeq(new_gray, 255);
figure("Name", "histeq2");
imshow(new_gray2);
figure("Name", "histeq2-hist10");
imhist(new_gray2, 10);
figure("Name", "histeq2-hist256");
imhist(new_gray2, 256);

%% 2.3 Linear Spatial Filtering

h1 = gaussian_filter(5, 5, 1);
h2 = gaussian_filter(5, 5, 2);
figure("Name", "filt1");
mesh(h1);
figure("Name", "filt2");
mesh(h2);

% gaussian noise image
img = imread("images/ntugn.jpg");
img = im2double(img);

filt1 = conv2(img, h1, "same");
filt2 = conv2(img, h2, "same");
figure("Name", "ntugn");
imshow(img);
figure("Name", "ntugn-h1");
imshow(filt1);
figure("Name", "ntugn-h2");
imshow(filt2);

% speckle noise image
img = imread("images/ntusp.jpg");
img = im2double(img);

filt1 = conv2(img, h1, "same");
filt2 = conv2(img, h2, "same");
figure("Name", "ntusp");
imshow(img);
figure("Name", "ntusp-h1");
imshow(filt1);
figure("Name", "ntusp-h2");
imshow(filt2);

%% 2.4 Median Filtering

% gaussian noise image
img = imread("images/ntugn.jpg");
img = im2double(img);

filt1 = medfilt2(img, [3, 3], "symmetric");
filt2 = medfilt2(img, [5, 5], "symmetric");
figure("Name", "ntugn");
imshow(img);
figure("Name", "ntugn-med3");
imshow(filt1);
figure("Name", "ntugn-med5");
imshow(filt2);

% speckle noise image
img = imread("images/ntusp.jpg");
img = im2double(img);

filt1 = medfilt2(img, [3, 3], "symmetric");
filt2 = medfilt2(img, [5, 5], "symmetric");
figure("Name", "ntusp");
imshow(img);
figure("Name", "ntusp-med3");
imshow(filt1);
figure("Name", "ntusp-med5");
imshow(filt2);

%% 2.5 Suppresing Noise Inteference Patterns

img = imread("images/pckint.jpg");
img = im2double(img);
figure("Name", "pckint");
imshow(img);

F = fft2(img);
S = abs(F);

figure("Name", "fourier-fftshift");
imagesc(fftshift(S .^ 0.1));
colormap("default");

figure("Name", "fourier");
imagesc(S .^ 0.1);
colormap("default");

x = [9, 249];
y = [241, 17];
new_F = F * 1;
new_F = mask_points(new_F, 5, x, y, 0);
new_S = abs(new_F);

figure("Name", "fourier-modified");
imagesc(fftshift(new_S .^ 0.1));
colormap("default");

new_img = ifft2(new_F);
new_img = im2uint8(new_img);
figure("Name", "pckint-modified");
imshow(new_img);

new_F2 = new_F * 1;
new_F2 = mask_vhlines(new_F2, x, y, 0);
new_S2 = abs(new_F2);

figure("Name", "fourier-enhanced");
imagesc(fftshift(new_S2 .^ 0.1));
colormap("default");

new_img2 = ifft2(new_F2);
new_img2 = im2uint8(new_img2);
figure("Name", "pckint-enhanced");
imshow(new_img2);

%% 2.5 (f)

img = imread("images/primatecaged.jpg");
img = rgb2gray(img);
img = im2double(img);
figure("Name", "primatecaged");
imshow(img);

F = fft2(img);
S = abs(F);

figure("Name", "fourier-fftshift");
imagesc(fftshift(S .^ 0.1));
colormap("default");

figure("Name", "fourier");
imagesc(S .^ 0.1);
colormap("default");

new_F = F * 1;

% main transform
x = [11, 22, 248, 237];
y = [252, 248, 4, 11];
new_F = mask_vhlines(new_F, x, y, 0);
new_F = mask_points(new_F, 5, x, y, 0);

% other transform
all_mul = [0.7, 0.95, 0.95];
all_thick = [3, 9, 19];
for i = 1:length(all_mul)
    mul = all_mul(i);
    thickness = all_thick(i);
    new_F = mask_line(new_F, 17, 4, 149, 240, thickness, mul);
    new_F = mask_line(new_F, 254, 241, 21, 104, thickness, mul);
    new_F = mask_line(new_F, 10, 30, 253, 245, thickness, mul);
    new_F = mask_line(new_F, 246, 225, 6, 18, thickness, mul);
    new_F = mask_line(new_F, 9, 38, 10, 27, thickness, mul);
    new_F = mask_line(new_F, 240, 213, 247, 229, thickness, mul);
    new_F = mask_line(new_F, 251, 205, 11, 36, thickness, mul);
    new_F = mask_line(new_F, 10, 48, 244, 223, thickness, mul);
end

new_S = abs(new_F);
figure("Name", "fourier-modified");
imagesc(fftshift(new_S .^ 0.1));
colormap("default");

new_img = ifft2(new_F);
new_img = im2uint8(new_img);
figure("Name", "primatecaged-modified");
imshow(new_img);

%% 2.6 Undoing Perspective Distortion of Planar Surface

img = imread('images/book.jpg');
figure("Name", "book");
imshow(img);

x = [143, 6, 257, 308];
y = [28, 159, 214, 47];
xn=[0, 0, 210, 210];
yn=[0, 297, 297, 0];

A = [
    x(1), y(1), 1, 0, 0, 0, -xn(1) * x(1), -xn(1) * y(1);
    0, 0, 0, x(1), y(1), 1, -yn(1) * x(1), -yn(1) * y(1);
    x(2), y(2), 1, 0, 0, 0, -xn(2) * x(2), -xn(2) * y(2);
    0, 0, 0, x(2), y(2), 1, -yn(2) * x(2), -yn(2) * y(2);
    x(3), y(3), 1, 0, 0, 0, -xn(3) * x(3), -xn(3) * y(3);
    0, 0, 0, x(3), y(3), 1, -yn(3) * x(3), -yn(3) * y(3);
    x(4), y(4), 1, 0, 0, 0, -xn(4) * x(4), -xn(4) * y(4);
    0, 0, 0, x(4), y(4), 1, -yn(4) * x(4), -yn(4) * y(4);
];
V = [0, 0, 0, 297, 210, 297, 210, 0]';
u = A \ V;
U = reshape([u; 1], 3, 3)'; 

w = U * [x; y; ones(1, 4)];  
w = w ./ (ones(3, 1) * w(3, :));
assert(isequal(round(w(1, :)), xn));
assert(isequal(round(w(2, :)), yn));

T= maketform("projective", U'); 
new_img = imtransform(img, T, 'XData', [0, 210], 'YData', [0, 297]);
figure("Name", "book-projected");
imshow(new_img);

%% Appendix: Functions

function new_img = contrast_stretching(img)
    img = im2double(img);
    min_val = min(img(:));
    max_val = max(img(:));
    new_img = (img - min_val) / max(max_val - min_val, eps);
    new_img = im2uint8(new_img);
end

function h = gaussian_filter(width, height, sigma)
    assert(mod(width, 2) == 1);
    assert(mod(height, 2) == 1);
    assert(sigma > 0);
    x = -1:2 / (width - 1):1;
    y = -1:2 / (height - 1):1;
    [X, Y] = meshgrid(x, y);
    h = (X .^ 2 + Y .^ 2) / (2 * sigma ^ 2);
    h = exp(-h) / (2 * sigma ^ 2 * pi);
    h = h / sum(h(:));
end

function F = mask_points(F, neigh, x, y, mul)
    assert(length(x) == length(y));
    height = size(F, 1);
    width = size(F, 2);
    offset = floor(neigh / 2);
    for i = 1:length(x)
        x1 = max(1, round(x(i)) - offset);
        x2 = min(width, round(x(i)) + offset);
        y1 = max(1, round(y(i)) - offset);
        y2 = min(height, round(y(i)) + offset);
        F(y1:y2, x1:x2) = mul * F(y1:y2, x1:x2);
    end
end

function F = mask_line(F, x1, x2, y1, y2, thickness, mul)
    height = size(F, 1);
    width = size(F, 2);
    space = max(abs(y1 - y2), abs(x1 - x2));
    x = x1:(x2 - x1) / (space - 1):x2;
    y = y1:(y2 - y1) / (space - 1):y2;
    offset = floor(thickness / 2);
    for i = 1:length(x)
        x1 = max(1, round(x(i)) - offset);
        x2 = min(width, round(x(i)) + offset);
        y1 = max(1, round(y(i)) - offset);
        y2 = min(height, round(y(i)) + offset);
        F(y1:y2, x1:x2) = F(y1:y2, x1:x2) * mul;
    end
end

function F = mask_vhlines(F, x, y, mul)
    for i = 1:length(x)
        F(:, x(i)) = F(:, x(i)) * mul;
    end
    for i = 1:length(y)
        F(y(i), :) = F(y(i), :) * mul;
    end
end
