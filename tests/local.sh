#!/bin/sh

# Test chip 

chip()
{   

    # Set vars
    passed=0
    width=65
    height=65
    expected=100
    in="tests/train/"
    out="tests/chips/"
    infile="test.tif"
    outfile="result_{}-result{}.tif"
    
    find $out -type f -name 'result*.tif' -delete
    python chip/chip.py $width $height $out $in $infile $outfile
    
    test=$(ls $out | grep 'result' | wc -l)
    find $out -type f -name 'result*.tif' -delete
    
    if [ $test == $expected ]
    then
        echo "The chipping module is working..."
    else
        echo "The chipping module is not working..."
    fi
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

mosaic()
{   
    passed=0
    chunksize="mask"
    extension="tests"
    root="flat" 
    img_type=".tif"
    
    python mosaic/mosaic.py $chunksize $extension $root $img_type
    
    if [ $test==$expected ]
    then
        echo "The mosaic module is working..."
    else
        echo "The mosaic module is not working..."
    fi
}
    
    
chip
mosaic
    
    