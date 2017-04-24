
loc=http://pages.cs.wisc.edu/~adbhat/

# grep '.tar' index.html | awk -F 'href=\"' '{print $2}' | awk -F '\"' '{print $1}' > remoteTarNames.txt

n1=n09246464_cliff_1648
n2=n09283405_foothill_1200
n3=n09303528_hillside_2011
n4=n09366317_naturalelevation_1534
n5=n09398677_precipice_1206
n6=n09399592_promontory_1543
n7=n09409752_ridge_1489

srcGrp=nature_hilly

mkdir -p $srcGrp
mkdir -p tars_${srcGrp}

for n in $n1 $n2 $n3 $n4 $n5 $n6 $n7
do
        mkdir -p ${srcGrp}/$n
        wget ${loc}${n}.tar
        tar -xvf ${n}.tar -C ${srcGrp}/$n
        mv ${n}.tar tars_${srcGrp}/
done


n1=n00433661_outdoorsport_1351
n2=n00439826_trackandfield_1849
n3=n00467719_sport_athleticgame_outdoorgame_fieldgame_1430
n4=n00523513_sport_1888
n5=n07942152_people_1432
n6=n09903153_celebrity_1316
n7=n10450303_politician_706

srcGrp=pplAndSports

mkdir -p $srcGrp

for n in $n1 $n2 $n3 $n4 $n5 $n6 $n7
do
        mkdir -p ${srcGrp}/$n
        wget ${loc}${n}.tar
        tar -xvf ${n}.tar -C ${srcGrp}/$n
        mv ${n}.tar tars_${srcGrp}/
done




