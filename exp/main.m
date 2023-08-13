function main
    
    Screen('Preference', 'SkipSyncTests', 1); 
    run_id =15;  %sssssss
    sub_name = 'heyaping';
    sub_id = 23;
    
    test_mode = 1;
    
    % trigger test 2
    % test or practice 16s
    % formal experiment 0
    
    stim_list_file = '2^4x5';
    
    port_id = 'D020';
    
    eye_to_screen = 751; % Eye-to-screen distance, millimeter
    window_size = 419; % Actual window size in width, millimeter
    display_size = 1024; % Actual window size in width, pixels
    
    switch test_mode
        case 0
            ioObj = io64;
            status = io64(ioObj); %#ok<*NASGU>
            address = hex2dec(port_id);
            Screen('Preference', 'SkipSyncTests', 0);
        case 1
            Screen('Preference', 'SkipSyncTests', 1);
        case 2
            ioObj = io64;
            status = io64(ioObj);
            address = hex2dec('D020');
            
            test_n = ['1000000'; '0100000'; '0010000'; '0001000'; ...
                '0000100'; '0000010'; '0000001'; '0000011'; '0000110'; ...
                '0001100'; '0011000'; '0110000'; '1100000'; '1110000'; ...
                '0111000'; '0011100'; '0001110'; '0000111'; '0001111'; ...
                '0011110'; '0111100'; '1111000'; '1111100'; '0111110'; ...
                '0011111'; '0111111'; '1111110'; '1111111'];
            
            test_n = bin2dec(test_n);
            
            for ii = 1:length(test_n)
                t = GetSecs;
                io64(ioObj, address, test_n(ii));
                while GetSecs - t < 0.01
                end
                io64(ioObj, address, 0);
                WaitSecs(0.5)
            end
            return
    end
    
    one_degree_pixs = round(eye_to_screen*tand(1)/window_size*display_size);
    scaling_factor = one_degree_pixs / 256;
    
    commandwindow;
    
    stimtable = readtable(stim_list_file, 'Delimiter', 'space', 'ReadVariableNames', 0);
    
    rep_n = 2;
    con_n = size(stimtable, 1);
    odd_n = con_n * rep_n * 0.1;
    exp_n = con_n * rep_n + odd_n;
    
    TABLE = zeros(exp_n, 4);
    % {'trial_id', 'is_back', 'stim_id', 'rt'}
    
    TABLE(:, 1) = (1:exp_n)';
    TABLE(DispersiveRandom(1:exp_n, odd_n), 2) = 1;
    TABLE(TABLE(:, 2) == 0, 3) = StimRandom(con_n*rep_n, con_n);
    TABLE(TABLE(:, 2) == 1, 3) = TABLE((find(TABLE(:, 2) == 1)-1), 3);
    
    Screen('Preference', 'VisualDebugLevel', 5);
    Screen('Preference', 'SuppressAllWarnings', 1);
    KbName('UnifyKeyNames');
    respKey = KbName('s');
    AssertOpenGL
    warning off %#ok<*WNOFF>
    HideCursor
    
    fix_img = imread('fix.png');
    bkg_img = imread('bkg.png');
    wptr = Screen('OpenWindow', 0, squeeze(bkg_img(1, 1, :)));
    ifi = 1 / FrameRate(wptr);
    
    scaled_rect = ScaleRect([0, 0, 2560, 2560], scaling_factor, scaling_factor);
    [x, y] = WindowCenter(wptr);
    destinationRect = CenterRectOnPointd(scaled_rect, x, y);
    
    for ii = 1:size(stimtable, 1)
        tmp_stim_path = sprintf('stim/%s%s%s%s/%d.png', ...
            stimtable.Var1{ii}, stimtable.Var2{ii}, stimtable.Var3{ii}, stimtable.Var4{ii}, ...
            stimtable.Var5(ii));
        tmp_image = imread(tmp_stim_path);
        stim_texture(ii) = Screen('MakeTexture', wptr, tmp_image); %#ok<*AGROW>
    end
    
    fix_texture = Screen('MakeTexture', wptr, fix_img);
    bkg_texture = Screen('MakeTexture', wptr, bkg_img);
    
    Screen('DrawTexture', wptr, fix_texture, [], destinationRect);
    Screen('Flip', wptr);
    
    while true
        [~, ~, key, ~] = KbCheck(-1);
        if key(respKey)
            break
        end
    end
    
    WaitSecs(1);
    
    for tr = 1:size(TABLE, 1)
        Screen('DrawTexture', wptr, stim_texture(TABLE(tr, 3)), [], destinationRect);
        vbl = Screen('Flip', wptr);
        if test_mode == 0
            if TABLE(tr, 2) == 1
                io64(ioObj, address, 99);
            else
                io64(ioObj, address, TABLE(tr, 3));
            end
            while GetSecs - vbl < 0.01
            end
            io64(ioObj, address, 0);
        end
        Screen('DrawTexture', wptr, fix_texture, [], destinationRect);
        start_time = Screen('Flip', wptr, vbl+0.2-0.33*ifi);
        keyPressed = 0;
        while (GetSecs - start_time) <= randi([900, 1100]) / 1000
            if ~keyPressed
                [~, end_time, key, ~] = KbCheck(-1);
                if key(respKey)
                    if test_mode == 0
                        io64(ioObj, address, 101);
                        while GetSecs - end_time < 0.01
                        end
                        io64(ioObj, address, 0);
                    end
                    TABLE(tr, 4) = (end_time - start_time) * 1000;
                    keyPressed = 1;
                elseif key(KbName('q'))
                    ShowCursor
                    sca
                    return
                end
            end
        end
    end
    
    WaitSecs(1);
    ShowCursor
    sca
    
    filename = sprintf('data/Subj%02d-Run%0d-%s-%s', ...
        sub_id, run_id, sub_name, datestr(now, 'yymmddHHMMSS'));
    
    hit_rate = length(find(TABLE(:, 2) == 1 & TABLE(:, 4) ~= 0)) / length(find(TABLE(:, 2) == 1));
    avg_rt = mean(TABLE(TABLE(:, 2) == 1 & TABLE(:, 4) ~= 0, 4));
    save(filename, 'TABLE', 'hit_rate', 'avg_rt');
    
    false_rate = length(find(TABLE(:, 2) == 0 & TABLE(:, 4) ~= 0)) / length(find(TABLE(:, 2) == 0));
    fprintf(' Hit  rate is %.2f.\n', hit_rate*100)
    fprintf('False rate is %.2f.\n', false_rate*100)
    fprintf('Average RT is %.2f.\n', avg_rt)
    
    function backIDVec = DispersiveRandom(inputVec, backNum)
        availBackID = floor(linspace(1, length(inputVec), backNum + 2));
        availBackID = availBackID(2:end-1);
        interval = availBackID(2) - availBackID(1);
        ratio = 0.3;
        availBackID = randi([-floor(interval * ratio), floor(interval * ratio)], ...
            1, length(availBackID)) + availBackID;
        backIDVec = inputVec(availBackID);
    end
    
    function stimMat = StimRandom(expN, conN)
        stimMat = [0, 0];
        while any(~diff(stimMat))
            stimMat = cell2mat(arrayfun(@randperm, repmat(conN, 1, expN / conN), ...
                'UniformOutput', false))';
        end
    end
end