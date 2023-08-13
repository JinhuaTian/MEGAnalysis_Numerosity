FSZ = [5.303, 7.500] / 2; % 半径(度)
ISZ = [0.191, 0.270] / 2; % 半径(度)
NUM = [16, 32]; % 数量 (偶数)

EXPMAT = CombVec(FSZ, ISZ, NUM)';

InputParser = {};
% Leave unchanged +++++++++
InputParser.image_size = 10; % 边长(度)
InputParser.eye_to_screen = 751; % 距离(毫米)
InputParser.winfows_size = 419; % 宽度(毫米) 390mm * 520mm
InputParser.display_size = 1024; % 宽度(像素) 1024 * 768
InputParser.background_color = [127, 127, 127]; % 背景颜色(RGB)
InputParser.black_dots_color = [0, 0, 0]; % 黑色圆点颜色(RGB)
InputParser.white_dots_color = [255, 255, 255]; % 白色圆点颜色(RGB)
InputParser.dotshape = 'fuckingass'; % square/circle/diamond/triangle/fuckingass

rng(19991028, 'v5uniform')

for REP = 1:10
    for idx = 1:length(EXPMAT)
        InputParser.field_size = EXPMAT(idx, 1);
        InputParser.item_size = EXPMAT(idx, 2);
        InputParser.item_number = EXPMAT(idx, 3);
        InputParser.item_spacing_thres = 0.25;
        image_name = sprintf('%d.png', REP);
        output_folder = sprintf('stimuli_%s/NM%dFA%dIA%d', ...
            InputParser.dotshape, ...
            find(NUM == InputParser.item_number), ...
            find(FSZ == InputParser.field_size), ...
            find(ISZ == InputParser.item_size));
        if ~exist(output_folder, 'dir')
            mkdir(output_folder)
        end
        InputParser.image_path = [output_folder, filesep, image_name];
        stim_one(InputParser)
    end
end

function stim_one(InputParser)
    %     Input parameters +++++++++
    eye_to_screen = InputParser.eye_to_screen;
    winfows_size = InputParser.winfows_size;
    display_size = InputParser.display_size;
    one_degree_pixs = round(eye_to_screen*tand(1)/winfows_size*display_size);
    one_degree_pixs = 256;
    
    image_size = InputParser.image_size;
    field_size = InputParser.field_size;
    item_size = InputParser.item_size;
    
    item_number = InputParser.item_number;
    
    item_spacing_thres = InputParser.item_spacing_thres;
    background_color = InputParser.background_color;
    black_dots_color = InputParser.black_dots_color;
    white_dots_color = InputParser.white_dots_color;
    dotshape = InputParser.dotshape;
    % Program begins +++++++++
    actual_image_size = 3001;
    centerxy = (actual_image_size + 1) / 2;
    image_size = round(image_size*one_degree_pixs);
    field_size = round(field_size*one_degree_pixs/image_size*actual_image_size);
    item_size = round(item_size*one_degree_pixs/image_size*actual_image_size);
    
    canvas = zeros(actual_image_size, actual_image_size, 3, 'uint8');
    [meshX, meshY] = meshgrid(1:length(canvas));
    
    mask = (meshX - centerxy).^2 + (meshY - centerxy).^2 <= field_size.^2;
    % mask: 0(black) - unavailable, 1(white) - available
    
    canvas(:, :, 1) = background_color(1);
    canvas(:, :, 2) = background_color(2);
    canvas(:, :, 3) = background_color(3);
    imwrite(imresize(canvas, [image_size, image_size]), 'blank.png')
    
    count = 1;
    
    s = RandStream('mlfg6331_64');
    
    if isequal(dotshape, 'fuckingass')
        if item_number == 16
            fucking_rotate_list = randsample(s, [0:15:105, 0:15:105], 16);
        elseif item_number == 32
            fucking_rotate_list = randsample(s, [0:15:105, 0:15:105, 0:15:105, 0:15:105], 32);
        end
    end
    
    while count <= item_number
        txy = randi([centerxy - (field_size - item_size), ...
            centerxy + (field_size - item_size)], [1, 2]);
        switch dotshape
            case 'circle'
                tcmask = (meshX - txy(1)).^2 + (meshY - txy(2)).^2 <= item_size.^2;
            case 'square'
                tcmask = abs(meshX-txy(1)+(meshY - txy(2))) + ...
                    abs(meshX-txy(1)-(meshY - txy(2))) <= item_size * sqrt(pi);
            case 'diamond'
                tcmask = abs(meshX-txy(1)) + abs(meshY-txy(2)) <= item_size / sqrt(2) * sqrt(pi);
            case 'triangle'
                tcmask1 = meshY - txy(2) <= -sqrt(3) * (meshX - txy(1)) + sqrt(3) * item_size * sqrt(pi/(3 * sqrt(3)));
                tcmask2 = meshY - txy(2) <= sqrt(3) * (meshX - txy(1)) + sqrt(3) * item_size * sqrt(pi/(3 * sqrt(3)));
                tcmask3 = meshY - txy(2) >= -item_size * sqrt(pi/(3 * sqrt(3)));
                tcmask = tcmask1 & tcmask2 & tcmask3;
            case 'fuckingass'
                this_angle = fucking_rotate_list(count);
                A = 0;
                B = 1;
                C = sqrt(pi/sqrt(27))*item_size-txy(2);
                [As, Bs, Cs] = FRotate(A, B, C, txy(1), txy(2), this_angle);
                tcmask1 = As*(meshX)+Bs*(meshY)+Cs>=0;
                [As, Bs, Cs] = FRotate(A, B, C, txy(1), txy(2), this_angle+120);
                tcmask2 = As*meshX+Bs*meshY+Cs>=0;
                [As, Bs, Cs] = FRotate(A, B, C, txy(1), txy(2), this_angle+240);
                tcmask3 = As*meshX+Bs*meshY+Cs>=0;
                tcmask = tcmask1&tcmask2&tcmask3;
        end
        
        tmask = (meshX - txy(1)).^2 + (meshY - txy(2)).^2 <= (item_size * (1 + item_spacing_thres)).^2;
        if isequal(tmask & mask, tmask)
            mask = mask - tmask;
            tmp_canvas = canvas(:, :, 1);
            if mod(count, 2)
                tmp_canvas(tcmask) = black_dots_color(1);
            else
                tmp_canvas(tcmask) = white_dots_color(1);
            end
            canvas(:, :, 1) = tmp_canvas;
            tmp_canvas = canvas(:, :, 2);
            if mod(count, 2)
                tmp_canvas(tcmask) = black_dots_color(2);
            else
                tmp_canvas(tcmask) = white_dots_color(2);
            end
            canvas(:, :, 2) = tmp_canvas;
            tmp_canvas = canvas(:, :, 3);
            if mod(count, 2)
                tmp_canvas(tcmask) = black_dots_color(3);
            else
                tmp_canvas(tcmask) = white_dots_color(3);
            end
            canvas(:, :, 3) = tmp_canvas;
            count = count + 1;
        end
    end
    
    % Save Dot Pattern  +++++++++
    canvas = imresize(canvas, [image_size, image_size]);
    imwrite(imresize(canvas, [image_size, image_size]), InputParser.image_path);
end

function [As, Bs, Cs] = FRotate(A, B, C, px, py, theta)
    Ts = inv([1, 0, px; 0, 1, py; 0, 0, 1] * ...
        [cosd(theta), -sind(theta), 0; ...
        sind(theta), cosd(theta), 0; ...
        0, 0, 1] * [1, 0, -px; 0, 1, -py; 0, 0, 1]);
    
    res = [A,B,C]*Ts;
    As = res(1);
    Bs = res(2);
    Cs = res(3);
end