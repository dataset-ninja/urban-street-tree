Authors introduce the **Tree** component for instance segmentation within The **Tree Dataset of Urban Street**, encompassing 3,949 high-resolution images distributed across 22 distinct classes (*acer_palmatum*, *cedrus_deodara* etc.). This specific section is designed to facilitate the precise delineation and identification of individual tree instances within the urban landscape. With these comprehensive resources at your disposal, this subset empowers researchers and practitioners to delve deep into the detailed analysis of urban street greenery, offering a valuable resource for comprehensive instance segmentation studies. Automatic tree species identification can be used to realize autonomous street tree inventories and help people without botanical knowledge and experience to better understand the diversity and regionalization of different urban landscapes.

## The Tree Dataset of Urban Street sub-datasets: 

### Classification:

- Branch 1485 images, 13 classes (1.4G) ([available on DatasetNinja](https://datasetninja.com/urban-street-branch))
- Trunk 7675 images, 29 classes (6.4G) ([available on DatasetNinja](https://datasetninja.com/urban-street-trunk))
- Leaf 21127 images, 50 classes (13.6G) ([available on DatasetNinja](https://datasetninja.com/urban-street-leaf-classification))
- Tree 4804 images, 23 classes (4.3G) ([available on DatasetNinja](https://datasetninja.com/urban-street-tree-classification))
- Fruit 4101 images, 29 classes (2.1G) ([available on DatasetNinja](https://datasetninja.com/urban-street-fruit))
- Flower 2275 images, 17 classes (1.3G) ([available on DatasetNinja](https://datasetninja.com/urban-street-flower))

### Segmentation:

- Tree 3949 images, 22 classes (7.9G) (current)
- Branch 1485 images, 13 classes (3.1G) ([available on DatasetNinja](https://datasetninja.com/urban-street-branch))
- Trunk 7675 images, 29 classes (12.9G) ([available on DatasetNinja](https://datasetninja.com/urban-street-trunk))
- Leaf 9763 images, 39 classes (10.2G) ([available on DatasetNinja](https://datasetninja.com/urban-street-leaf))

### Detection:

- Leaf 9763 images, 39 classes (11G) ([available on DatasetNinja](https://datasetninja.com/urban-street-leaf))

<img src="https://ytt917251944.github.io/dataset_jekyll/assets/img/seg/segmentation-tree.png" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Examples of Urban Street: Tree (segmenation task).</span>

## About Tree Dataset of Urban Street

Annotations were performed in a fine-grained manner by using polygons (bitmap in supervisely) to outline individual objects. Authors assessed the performance of various vision algorithms on different classification and segmentation tasks, including tree species identification and instance segmentation. 

The proposed dataset was designed to capture urban street trees with subtropical or temperate monsoon climates in China. Our data collection and annotation methods were carefully created to capture the high variability of street trees. From February to October 2022, tens of thousands of tree images were acquired with mobile devices, covering spring, summer, fall and winter in 10 cities.

Similar to Cityscapes (Cordts et al., 2016) ([available on DatasetNinja](https://datasetninja.com/cityscapes)) and ADE20K (Zhou et al., 2019) ([available on DatasetNinja](https://datasetninja.com/ade20k)), authors divide each organ dataset into separate training (*train*), validation (*val*) and test (*test*) sets. 
