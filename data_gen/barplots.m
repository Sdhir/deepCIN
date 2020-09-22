
close all;
clear;
%att = [0.05830723 0.05787396 0.0585538  0.0570809  0.05997266 0.06448789 0.06304483 0.05982037 0.05884344 0.0588302  0.05763124 0.0587697 0.05630854 0.05740373 0.05781671 0.05758506 0.05766974];
att=[0.14328997 0.1432953  0.14458965 0.14283523 0.14223893 0.14129907 0.14245187]
b = bar(att,'BarWidth', 1); hold on;
% caxis([min(att) max(att)])
% colorbar

ylim([min(att)-0.1*(max(att)-min(att)), max(att)+0.1*(max(att)-min(att))]);
xlim([0,length(att)+1]);
xticks(1:length(att));
ylabel('Probability','FontSize',25)
xlabel('Segments','FontSize',25)
ax = gca;
ax.FontSize = 20; 
set(gcf,'position',[10,10,750,400])
b.FaceColor = 'flat';
for i=1:length(att)
    if att(i)>mean(att)
        b.CData(i,:) = [0 0.8 .8];
        %text(i,att(i),num2str(att(i),'%.2g'),'vert','bottom','horiz','center'); 
    end
end

plot(xlim,[mean(att) mean(att)], '--k');