function plotNavalDS(t,data,labels)

figure()
hold on
for k = 1:size(data,1)
    
%     if min(min(min(data(k,2,:)))) < 20
%         color = 'r';
%         labels(k) = -1;
%     elseif data(k,1,end) > 50
%         color = 'k';
%         labels(k) = -1;
%     else
%         color = 'g';
%     end
    
    if labels(k) == -1
        color = 'r';
    else
        color = 'g';
    end
    plot(reshape(data(k,1,:),length(t),1),reshape(data(k,2,:),length(t),1),color);
end

axis([0 80 15 45])
xlabel ('x (dam)') 
ylabel ('y (dam)')
title('Naval scenario')

end
