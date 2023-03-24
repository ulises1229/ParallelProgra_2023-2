#include <iostream>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"// read pictures stb library
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // write prictures stb library
// Soruce: https://github.com/nothings/st

#define CHANNEL_NUM 3

using namespace std;


uint8_t* readImage(char* file, int &width, int &height, int bpp);
void writeImage(char* file, int width, int height, uint8_t *image);


int main() {
    // va
    int width, height, bpp;
    char *inFile= "/tmp/tmp.J6koHzOysZ/figs/lena.jpg"; // TODO: change this path as required

    // read an RGB image
    uint8_t * image = readImage(inFile, width, height, bpp);

    char *outImage= "/tmp/tmp.J6koHzOysZ/figs/test.jpg"; // TODO: change this path as required
    // write an RGB image
    writeImage(outImage, width, height, image);


    return 0;
}


uint8_t* readImage(char *file, int &width, int &height, int bpp){
    //
    uint8_t *rgb_image = stbi_load(file, &width, &height, &bpp, CHANNEL_NUM);
    cout<< "Image size: " << width << " x " << height  << " = " << width * height  << " pixels"<< endl;
    return rgb_image;
}

void writeImage(char* file, int width, int height,  uint8_t *image){
    stbi_write_png(file, width, height, CHANNEL_NUM, image, width*CHANNEL_NUM);

}

