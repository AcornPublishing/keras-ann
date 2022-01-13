speed = 1000:1000:15000;
torque = 0:5:50;

[S, T] = meshgrid(speed, torque);

data = csvread('max_eff_id.csv', 1, 1);

for i=2:11
    for j = 2:15
        if data(i, j) == 0
            data(i, j) = inf;
        end
    end
end

figure(1);
mesh(S, T, data);
view(120, 20);
title('Id');
xlabel('Speed [r/min]');
ylabel('Torque [Nm]');
zlabel('Id [A]');
set(gca, 'fontsize', 12);

data = csvread('max_eff_iq.csv', 1, 1);

for i=2:11
    for j = 2:15
        if data(i, j) == 0
            data(i, j) = inf;
        end
    end
end

figure(2);
mesh(S, T, data);
view(30, 20);
title('Iq');
xlabel('Speed [r/min]');
ylabel('Torque [Nm]');
zlabel('Iq [A]');
set(gca, 'fontsize', 12);