#!/bin/sh

# Test chip 

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
    passed=0
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
    storage ="mask"
    mode="standard" 
    extension=".tif" 
    shape_type="shape"
    image_type="train"  
    shape_root="tests" 
    output_format=".tif" 
    shape_name="vegas.geojson"
    
    python mask/mask.py $root $image_type $shape_root \
    $output_format $shape_type $shape_name $mode $extension $storage
    
     
}
   

flatten()
{   
    passed=0
    mask="mask"
    root="tests"
    storage="flat" 
    extension=".tif"
    image_type="train" 
    output_format=".pkl" 
    
    python flatten/flatten.py $output_format $root $image_type $extension $storage $mask
    
    if [ $test==$expected ]
    then
        echo "The flatten module is working..."
    else
        echo "The flatten module is not working..."
    fi
}


split()

{
    
    echo "Split will go here..."

}


summarize()
{


    echo "Summarize will go here..."

}

 
run()    
{   
    chip
    mask
    mosaic
    
    chip
    flatten
    
    chip
    split
    summarize
    
}


run
    