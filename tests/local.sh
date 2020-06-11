#!/bin/sh


chip()
{   
    
    width=65
    height=65
    expected=100
    in="tests/train/"
    out="tests/chip/"
    infile="test.tif"
    outfile="result_{}-result{}.tif"
    
    find $out -type f -name 'result*.tif' -delete
    
    python chip/chip.py $width $height $out $in $infile $outfile
    test=$(ls $out | grep 'result' | wc -l)
    
    if [ $test == $expected ]
    then
        echo "The chipping module is working..."
    else
        echo "The chipping module is not working..."
    fi
}

mosaic()
{   
    root="tests" 
    chunksize=100
    img_type="chip"
    extension=".tif"
    out_fp="mosaic"
    
    in="tests/train/"
    out="tests/chip/"
    infile="test.tif"
    
    python mosaic/mosaic.py $chunksize $extension $root $img_type $out_fp
    
    find $out -type f -name 'result*.tif' -delete
    diff "$root/$out_fp/chunk_0.tif" "$root/$in/$infile"
    
    
}

mask()
{
    root="tests"
    storage="mask"
    mode="standard" 
    extension=".tif" 
    shape_type="shape"
    image_type="train"  
    shape_root="tests" 
    output_format=".tif" 
    shape_name="vegas.geojson"
    
    python mask/mask.py $root $image_type $shape_root \
    $output_format $shape_type $shape_name $mode $extension $storage
    
    diff "$root/$storage/mask.tif" "$root/$storage/test.tif"
    find "$root/$storage" -type f -name 'test.tif' -delete
     
}
   

flatten()
{   
    mask="mask"
    root="tests"
    storage="flat" 
    extension=".tif"
    image_type="train" 
    output_format=".npz" 
    
    python flatten/flatten.py $output_format $root $image_type $extension $storage $mask
    
}


split()

{
    expected=75
    root="tests"
    image_type="chip"
    train_split=0.75
    val_target="val_frames"
    train_target="train_frames"
    
    python split/split.py $root $image_type $train_split $train_target $val_target
    
    train=$(ls $root/$train_target | grep 'result' | wc -l)
    val==$(ls $root/$val_target | grep 'result' | wc -l)
    
    if [ train==$expected ]
    then 
        echo "The split module is working fine..."
    else
        echo "Check the split module..."
    fi
    
    find $root/$image_type -type f -name 'result*.tif' -delete
    rm -r $root/$train_target
    rm -r $root/$val_target
        
}


summarize()
{

    echo '.....'

}

 
run()    
{   
flatten
    
}


run