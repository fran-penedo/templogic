%% Create Test Traces
Nobj=300;

t = 0:1:99;

rect1 = 7*rectpuls(t-40,10);
rect2 = 6*rectpuls(t-70,10);
data_norm = 2*ones(Nobj,100)+repmat(rect1,Nobj,1)+repmat(rect2,Nobj,1);
labels_norm(1:Nobj) = 1;

rect3 = 6*rectpuls(t-40,10);
rect4 = 7*rectpuls(t-70,10);

data_anom = 2*ones(Nobj,100)+repmat(rect3,Nobj,1)+repmat(rect4,Nobj,1);
labels_anom(1:Nobj) = -1;

data_tmp = [data_norm; data_anom];
labels = [labels_norm, labels_anom];

I = randperm(size(data_tmp,1));
data_tmp = data_tmp(I,:);
labels = labels(I);

noise = 0 + 0.5.*randn(Nobj*2,100);
data_tmp = data_tmp+noise;

data = zeros(Nobj*2,1,100);
data(:,1,:) = data_tmp;

%plotTLIsignals(t,data,labels)

hold on
for i = 1:length(t)
    if labels(i)> 0
        plot(t, reshape(data(i, 1, :), [100, 1]), 'b')
    else
        plot(t, reshape(data(i, 1, :), [100, 1]), 'r')
    end
end
hold off

save('simpleDS2','data','labels','t');