% % Input parameters
% orders = -2:4;
% theta_i_values = 45;
% lambda = 0.905; % Wavelength in micrometers
% pitch = 13.68 / sqrt(2); % Pitch in micrometers
% sensor_size = 3.2; % Sensor size in mm
% focal_lengths = 24.5; % Two focal lengths in mm
% 
% % Initialize arrays to store results for both focal lengths
% all_theta_d = zeros(length(focal_lengths), length(orders));
% all_sub_fov = zeros(length(focal_lengths), length(orders));
% 
% % Plot diffraction angles vs. diffraction orders for all incident angles
% figure;
% hold on;
% 
% % Loop through focal length
% for f = 1:length(focal_lengths)
%     focal_length = focal_lengths(f);
% 
%     for i = 1:length(theta_i_values)
%         theta_i = theta_i_values(i);
% 
%         % Calculate diffraction angles
%         theta_d = asind(sind(theta_i) - orders * lambda / pitch);
%         all_theta_d(f, :) = theta_d; 
% 
%         % Calculate FOV at 0th order (angular)
%         fov_0_angle = 2 * atand(sensor_size / (2 * focal_length)); % FOV angle in degrees
% 
%         % Calculate chief ray angles
%         chief_1 = theta_i - fov_0_angle / 2;
%         chief_2 = theta_i + fov_0_angle / 2;
% 
%         % Calculate sub-FOVs (in degrees)
%         sub_fov = asind(sind(chief_2) - orders * lambda / pitch) - asind(sind(chief_1) - orders * lambda / pitch);
%         all_sub_fov(f, :) = sub_fov; 
%     end
% 
%     % Calculate error bars
%     error_bars = all_sub_fov(f, :) / 2; 
% 
%     % Plot with error bars - different colors for different focal lengths
%     if f == 1
%         color = 'r'; % Red for 24mm
%     else
%         color = 'b'; % Blue for 72mm
%     end
% 
%     % Plot with error bars
%     errorbar(orders, all_theta_d(f, :), error_bars, 'o-', 'LineWidth', 1.5, ...
%         'Color', color, 'DisplayName', ['$\theta_i = ' num2str(theta_i_values(1)) '^\circ$ (f = ' num2str(focal_lengths(f)) 'mm)']);
% 
%     % Add horizontal dashed lines at top and bottom of error bars
%     for j = 1:length(orders)
%         y_top = all_theta_d(f, j) + error_bars(j);
%         y_bottom = all_theta_d(f, j) - error_bars(j);
% 
%         % Horizontal dashed lines
%         yline(y_top, '--', 'Color', color, 'LineWidth', 1, HandleVisibility='off');
%         yline(y_bottom, '--', 'Color', color, 'LineWidth', 1, HandleVisibility='off');
% 
%         % Annotate data points with values with different vertical offsets
%         if f == 1  % 24mm focal length (text below)
%             text(orders(j), all_theta_d(f, j), ...
%                  sprintf('%.2f ± %.2f\n\n', all_theta_d(f, j), error_bars(j)), ...
%                  'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
%                  'FontSize', 10, 'Color', 'r');
%         else       % 72mm focal length (text above)
%             text(orders(j), all_theta_d(f, j), ...
%                  sprintf('\n\n%.2f ± %.2f', all_theta_d(f, j), error_bars(j)), ...
%                  'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
%                  'FontSize', 10, 'Color', 'b');
%         end
%     end
% end
% 
% xlim([-2.5 4.5]); 
% xlabel('Diffraction Order', 'FontSize', 15);
% ylabel('Diffraction Angle (degrees)', 'FontSize', 15);
% title('RxDMD Steering Angles vs. Diffraction Orders', 'FontSize', 15);
% legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', 15);
% grid on;
% hold off;


%----For Single Lens------%
% Input parameters
orders = -2:4;
theta_i = 45; % Incident angle in degrees
lambda = 0.905; % Wavelength in micrometers
pitch = 13.68 / sqrt(2); % Pitch in micrometers
sensor_size = 3.2; % Sensor size in mm
focal_length = 24.13; % Single focal length in mm
offset = 5.5;
% Initialize arrays to store results
theta_d = asind(sind(theta_i) - orders * lambda / pitch);

% Calculate FOV at 0th order (angular)
fov_0_angle = 2 * atand(sensor_size / (2 * focal_length)); % FOV angle in degrees

% Calculate chief ray angles
chief_1 = theta_i - fov_0_angle / 2;
chief_2 = theta_i + fov_0_angle / 2;

% Calculate sub-FOVs (in degrees)
sub_fov = asind(sind(chief_2) - orders * lambda / pitch) - asind(sind(chief_1) - orders * lambda / pitch);

% Calculate error bars (half-FOV)
error_bars = sub_fov / 2;

% Plot diffraction angles vs. diffraction orders
figure;
hold on;

% Plot with error bars (legend only for main plot)
errorbar(orders, theta_d, error_bars, 'o-', 'LineWidth', 5, ...
    'Color', 'b', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'k', ...
    'DisplayName', ['$\theta_i = ' num2str(theta_i) '^\circ$ (f = ' num2str(focal_length) 'mm)']);

% Add horizontal dashed lines at top and bottom of error bars
for j = 1:length(orders)
    y_top = theta_d(j) + error_bars(j);
    y_bottom = theta_d(j) - error_bars(j);
    
    % Thick green half-FOV lines (error bars) without legend
    plot([orders(j), orders(j)], [y_bottom, y_top], 'g', 'LineWidth', 2, 'HandleVisibility', 'off');
    
    % Black dashed lines at top and bottom (without legend)
    yline(y_top, '--k', 'LineWidth', 1, 'HandleVisibility', 'off');
    yline(y_bottom, '--k', 'LineWidth', 1, 'HandleVisibility', 'off');

    % Annotate data points with values
    text(orders(j), theta_d(j)-offset, ...
         sprintf('%.2f ± %.2f', theta_d(j), error_bars(j)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
         'FontSize', 15, 'Color', 'r');
end

xlim([-2.5 4.5]); 
xlabel('Diffraction Orders', 'FontSize', 20);
ylabel('Diffraction Angle (degrees)', 'FontSize', 20);
title('RxDMD Steering Angles vs. Diffraction Orders', 'FontSize', 25);
legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', 20);
grid on;
hold off;
