import numpy as np
import cv2


def get_limits(color):
    # Convert RGB to BGR
    color_bgr = color[::-1]  # Reverse the order from RGB to BGR

    c = np.uint8([[color_bgr]])  # Insert BGR values that you want to convert to HSV
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]

    lowerLimit = (max(hue - 10, 0), 100, 100)
    upperLimit = (min(hue + 10, 179), 255, 255)

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit

def get_color_bgr_values():
    return {
    'Red': [0, 0, 255],
    'Green': [0, 255, 0],
    'Blue': [255, 0, 0],
    'Yellow': [0, 255, 255],
    'Cyan': [255, 255, 0],
    'Magenta': [255, 0, 255],
    'White': [255, 255, 255],
    'Black': [0, 0, 0],
    'Gray': [128, 128, 128],
    'Orange': [0, 165, 255],
    'Pink': [203, 192, 255],
    'Purple': [128, 0, 128],
    'Brown': [42, 42, 165],
    'Maroon': [0, 0, 128],
    'Olive': [0, 128, 128],
    'Lime': [0, 255, 0],
    'Teal': [128, 128, 0],
    'Navy': [128, 0, 0],
    'Aquamarine': [212, 255, 127],
    'Azure': [240, 255, 255],
    'Beige': [220, 245, 245],
    'Bisque': [196, 228, 255],
    'Chocolate': [30, 105, 210],
    'Coral': [80, 127, 255],
    'Crimson': [60, 20, 220],
    'Fuchsia': [255, 0, 255],
    'Gold': [0, 215, 255],
    'Ivory': [240, 255, 255],
    'Khaki': [140, 230, 240],
    'Lavender': [230, 230, 250],
    'Linen': [250, 240, 230],
    'Moccasin': [181, 228, 255],
    'Orchid': [214, 112, 218],
    'Peru': [63, 133, 205],
    'Plum': [221, 160, 221],
    'Salmon': [114, 128, 250],
    'Sienna': [45, 82, 160],
    'Silver': [192, 192, 192],
    'Tan': [210, 180, 140],
    'Thistle': [216, 191, 216],
    'Tomato': [71, 99, 255],
    'Turquoise': [208, 224, 64],
    'Violet': [238, 130, 238],
    'Wheat': [179, 222, 245],
    'Sky Blue': [235, 206, 135],
    'Forest Green': [34, 139, 34],
    'Indian Red': [92, 92, 205],
    'Steel Blue': [180, 130, 70],
    'Dark Slate Blue': [139, 61, 72],
    'Medium Spring Green': [0, 250, 154],
    'Rosy Brown': [188, 143, 143],
    'Dark Olive Green': [85, 107, 47],
    'Saddle Brown': [19, 69, 139],
    'Medium Purple': [147, 112, 219],
    'Medium Sea Green': [60, 179, 113],
    'Fire Brick': [34, 34, 178],
    'Cadet Blue': [160, 158, 95],
    'Pale Violet Red': [147, 112, 219],
    'Dark Khaki': [107, 183, 189],
    'Light Coral': [128, 128, 240],
    'Dark Salmon': [122, 150, 233],
    'Medium Aquamarine': [102, 205, 170],
    'Dark Orchid': [153, 50, 204],
    'Medium Orchid': [186, 85, 211],
    'Dark Turquoise': [0, 206, 209],
    'Medium Turquoise': [72, 209, 204],
    'Light Salmon': [122, 160, 255],
    'Dark Goldenrod': [11, 134, 184],
    'Medium Violet Red': [133, 21, 199],
    'Light Green': [144, 238, 144],
    'Light Sky Blue': [250, 206, 135],
    'Light Steel Blue': [222, 196, 176],
    'Pale Goldenrod': [170, 232, 238],
    'Light Yellow': [224, 255, 255],
    'Mint Cream': [250, 255, 250],
    'Honeydew': [240, 255, 240],
    'Alice Blue': [240, 248, 255],
    'Ghost White': [240, 255, 255],
    'Snow': [250, 250, 255],
    'Seashell': [255, 245, 238],
    'Floral White': [255, 250, 240],
    'Old Lace': [253, 245, 230],
    'Antique White': [250, 235, 215],
    'Papaya Whip': [255, 239, 213],
    'Blanched Almond': [255, 235, 205],
    'Lavender Blush': [255, 240, 245],
    'Cornsilk': [255, 248, 220],
    'Lemon Chiffon': [255, 250, 205],
    'Light Goldenrod Yellow': [250, 250, 210],
    'Light Cyan': [224, 255, 255],
    'Pale Green': [152, 251, 152],
    'Spring Green': [0, 255, 127],
    'Sea Green': [46, 139, 87],
    'Dark Green': [0, 100, 0],
    'Yellow Green': [50, 205, 154],
    'Olive Drab': [35, 142, 107],
    'Dark Sea Green': [143, 188, 143],
    'Light Sea Green': [170, 178, 32],
    'Light Goldenrod': [85, 236, 250],
    'Goldenrod': [32, 165, 218],
    'Burlywood': [222, 184, 135],
    'Navajo White': [255, 222, 173],
    'Peach Puff': [255, 218, 185],
    'Light Pink': [255, 182, 193],
    'Deep Pink': [255, 20, 147],
    'Hot Pink': [255, 105, 180],
    'Violet Red': [255, 35, 85],
    'Medium Slate Blue': [123, 104, 238],
    'Cornflower Blue': [100, 149, 237],
    'Royal Blue': [65, 105, 225],
    'Dodger Blue': [30, 144, 255],
    'Deep Sky Blue': [0, 191, 255],
    'Light Blue': [173, 216, 230],
    'Powder Blue': [176, 224, 230],
    'Pale Turquoise': [175, 238, 238],
    'Peach Puff': [255, 218, 185],
    'Pale Goldenrod': [238, 232, 170],
    'Pale Turquoise': [175, 238, 238],
    'Charcoal': [54, 69, 79],
    'Dark Gray': [169, 169, 169],
    'Light Gray': [211, 211, 211],
    'Ash Gray': [178, 190, 181],
    'Jet': [52, 52, 52],
    'Smoky Black': [8, 12, 16],
    'Outer Space': [76, 74, 65],
    'Timberwolf': [210, 215, 219],
    'Platinum': [226, 228, 229],
    'Old Silver': [132, 132, 130],
    'Khaki Drab': [51, 60, 69],
    'Desert Sand': [44, 93, 102],
    'Pale Silver': [187, 192, 201],
    'Shadow': [93, 121, 138],
    'Pewter Blue': [183, 168, 139],
    'Cool Black': [99, 46, 0],
    'Cerulean': [208, 189, 163],
    'Ultramarine': [143, 10, 18],
    'Electric Blue': [0, 255, 204],
    'Jade': [107, 168, 0],
    'Persian Blue': [204, 48, 54],
    'Cerulean Blue': [205, 92, 92],
    'Egyptian Blue': [166, 52, 16],
    'Klein Blue': [255, 48, 30],
    'Space Cadet': [102, 57, 176],
    'Midnight': [25, 25, 112],
    'Gun metal': [42, 52, 57],
    'Black Olive': [59, 60, 54],
    'Davys Grey': [85, 85, 85],
    'Raisin Black': [36, 33, 36],
    'Eerie Black': [27, 27, 27],
    'Taupe': [72, 60, 50],
    'Coal': [54, 54, 54],
    'Licorice': [26, 17, 16],
    'Rich Black': [1, 11, 19],
    'Vantablack': [0, 0, 0],
    'Iridium': [55, 55, 55],
    'Obsidian': [21, 22, 31],
    'Apple Green': [0, 182, 141],
    'Apricot': [177, 206, 251],
    'Aqua': [255, 255, 0],
    'Army Green': [32, 83, 75],
    'Asparagus': [107, 169, 135],
    'Atomic Tangerine': [102, 153, 255],
    'Baker-Miller Pink': [175, 145, 255],
    'Banana Yellow': [53, 225, 255],
    'Barn Red': [2, 10, 124],
    'Battleship Grey': [130, 132, 132],
    'Bazaar': [123, 119, 152],
    'Blizzard Blue': [238, 229, 172],
    'Blue Bell': [208, 162, 162],
    'Blue Green': [186, 152, 13],
    'Blue Violet': [226, 43, 138],
    'Blush': [131, 93, 222],
    'Bole': [59, 68, 121],
    'Brick Red': [84, 65, 203],
    'Bright Green': [0, 255, 102],
    'Bright Lilac': [239, 145, 216],
    'Bright Maroon': [72, 33, 195],
    'Bright Navy Blue': [210, 116, 25],
    'Bright Pink': [127, 0, 255],
    'Bright Turquoise': [222, 232, 8],
    'Bright Ube': [232, 159, 209],
    'Bright Yellow': [0, 255, 255],
    'Brilliant Rose': [163, 85, 255],
    'Brink Pink': [127, 96, 251],
    'British Racing Green': [37, 66, 0],
    'Bronze': [50, 127, 205],
    'Brown Sugar': [77, 110, 175],
    'Bud Green': [97, 182, 123],
    'Buff': [130, 220, 240],
    'Burgundy': [32, 0, 128],
    'Byzantine': [164, 51, 189],
    'Byzantium': [99, 41, 112],
    'Cambridge Blue': [173, 193, 163],
    'Camel': [107, 154, 193],
    'Capri': [255, 191, 0],
    'Cardinal': [58, 30, 196],
    'Caribbean Green': [153, 204, 0],
    'Carmine': [24, 0, 150],
    'Carnation Pink': [201, 166, 255],
    'Carnelian': [27, 27, 179],
    'Carrot Orange': [33, 145, 237],
    'Ceil': [207, 161, 146],
    'Celadon': [175, 225, 172],
    'Celeste': [255, 255, 178],
    'Cerise': [99, 49, 222],
    'Cerulean Frost': [195, 155, 109],
    'Champagne': [206, 231, 247],
    'Cherry': [99, 49, 222],
    'Chestnut': [53, 69, 149],
    'China Pink': [161, 111, 222],
    'Chinese Red': [30, 56, 170],
    'Chocolate (Traditional)': [0, 63, 123],
    'Cinereous': [123, 129, 152],
    'Cinnabar': [52, 66, 227],
    'Cinnamon': [30, 105, 210],
    'Citrine': [10, 208, 228],
    'Classic Rose': [231, 204, 251],
    'Cobalt Blue': [171, 71, 0],
    'Columbia Blue': [255, 221, 155],
    'Congo Pink': [121, 131, 248],
    'Copper': [51, 115, 184],
    'Copper Rose': [102, 102, 153],
    'Coquelicot': [0, 56, 255],
    'Coral Pink': [121, 131, 248],
    'Cordovan': [69, 63, 137],
    'Cornflower': [237, 149, 100],
    'Cosmic Cobalt': [136, 45, 46],
    'Cosmic Latte': [231, 248, 255],
    'Cotton Candy': [217, 188, 255],
    'Cream': [208, 253, 255],
    'Crimson Red': [0, 0, 153],
    'Cyan Blue': [235, 183, 0],
    'Cyber Grape': [124, 66, 88],
    'Cyber Yellow': [0, 211, 255],
    'Daffodil': [49, 255, 255],
    'Dandelion': [48, 225, 240],
    'Dark Brown': [33, 67, 101],
    'Dark Candy Apple Red': [0, 0, 164],
    'Dark Cerulean': [126, 69, 8],
    'Dark Chestnut': [96, 105, 152],
    'Dark Coral': [69, 91, 205],
    'Dark Cyan': [139, 139, 0],
    'Dark Electric Blue': [120, 104, 83],
    'Dark Imperial Blue': [106, 65, 0],
    'Dark Jungle Green': [33, 36, 26],
    'Dark Lava': [50, 60, 72],
    'Dark Lavender': [150, 79, 115],
    'Dark Liver': [79, 75, 83],
    'Dark Magenta': [139, 0, 139],
    'Dark Medium Gray': [169, 169, 169],
    'Dark Midnight Blue': [102, 51, 0],
    'Dark Moss Green': [35, 93, 74],
    'Dark Pastel Green': [60, 192, 3],
    'Dark Red': [0, 0, 139],
    'Dark Scarlet': [25, 3, 86],
    'Dark Sienna': [20, 20, 60],
    'Dark Spring Green': [69, 114, 23],
    'Dark Tan': [81, 129, 145],
    'Dark Tangerine': [18, 168, 255],
    'Dark Terracotta': [92, 78, 204],
    'Dark Violet': [211, 0, 148],
    'Jet Black': [0, 0, 0],
    'Ebony': [55, 71, 79],
    'Onyx': [15, 15, 15],
    'Obsidian': [21, 22, 31],
    'Dark Lava': [72, 60, 50],
}


def get_color_objects():
    return {
        'Red': ['Apple', 'Tomato', 'Firetruck'],
        'Green': ['Leaf', 'Frog', 'Grass'],
        'Blue': ['Sky', 'Ocean', 'Blueberries'],
        'Yellow': ['Banana', 'Sunflower', 'Lemon'],
        'Cyan': ['Tropical Waters', 'Blue Jay Feathers', 'Swimming Pool'],
        'Magenta': ['Raspberries', 'Orchid', 'Beet Juice'],
        'White': ['Snow', 'Cloud', 'Cotton'],
        'Black': ['Coal', 'Obsidian', 'Night'],
        'Gray': ['Stone', 'Ash'],
        'Orange': ['Orange', 'Carrot', 'Pumpkin'],
        'Pink': ['Flamingo', 'Cotton Candy', 'Rose'],
        'Purple': ['Eggplant', 'Grapes', 'Plums'],
        'Brown': ['Chocolate', 'Wood', 'Coffee'],
        'Maroon': ['Red wine', 'Leather', 'Pomegranate'],
        'Olive': ['Olives', 'Moss'],
        'Lime': ['Limes', 'Green Apples'],
        'Teal': ['Kingfisher Bird'],
        'Navy': ['Blue Pen Ink', 'Midnight Sky'],
        'Aquamarine': ['Sea Glass', 'Ocean Water', 'Gemstones'],
        'Azure': ['Clear Sky', 'Peacock', 'Blue Jays'],
        'Beige': ['Sand', 'Khaki'],
        'Bisque': ['Porcelain Dolls', 'Eggshells'],
        'Chocolate': ['Milk Chocolate', 'Chocolate Cake'],
        'Coral': ['Coral Reef', 'Tropical Fish'],
        'Crimson': ['Dark Rose', 'Blood'],
        'Fuchsia': ['Dragon fruit', 'Butterflies'],
        'Gold': ['Gold Nugget', 'Coins', 'Rings'],
        'Ivory': ['Elephant Tusk', 'Animal Teeth'],
        'Khaki': ['Sand', 'Mushrooms', 'Camel Fur'],
        'Lavender': ['Lavender Flower', 'Sunsets'],
        'Linen': ['Cloth', 'Bedsheets', 'Napkins'],
        'Moccasin': ['Fur Slippers'],
        'Orchid': ['Orchids'],
        'Peru': ['Autumn Leaves', 'Rust', 'Brown Sugar'],
        'Plum': ['Plums', 'Eggplant', 'Beetroot'],
        'Salmon': ['Salmon', 'Sunsets', 'Flamingo'],
        'Sienna': ['Bricks', 'Dried Leaves'],
        'Silver': ['Quarter', 'Nickel'],
        'Tan': ['Sand', 'Leather', 'Birch'],
        'Thistle': ['Thistle Flower'],
        'Tomato': ['Tomato'],
        'Turquoise': ['Gem', 'Parrots'],
        'Violet': ['Violet'],
        'Wheat': ['Wheat'],
        'Sky Blue': ['Sky'],
        'Forest Green': ['Pine Tree'],
        'Indian Red': ['Indian Sunset'],
        'Steel Blue': ['Blue Heron Feathers', 'Blueberries'],
        'Dark Slate Blue': ['Slate Stone'],
        'Medium Spring Green': ['Spring Grass'],
        'Rosy Brown': ['Rosy Cheeks'],
        'Dark Olive Green': ['Olive Tree', 'Olives'],
        'Saddle Brown': ['Horse Fur', 'Chestnuts'],
        'Medium Purple': ['Purple Iris'],
        'Medium Sea Green': ['Seaweed'],
        'Fire Brick': ['Brick Wall'],
        'Cadet Blue': ['Blue Jay Feathers', 'Gray Whale Skin'],
        'Pale Violet Red': ['Violet Rose'],
        'Dark Khaki': ['Khaki Jacket'],
        'Light Coral': ['Coral Sand'],
        'Dark Salmon': ['Salmon Fillet'],
        'Medium Aquamarine': ['Aquamarine Gem'],
        'Dark Orchid': ['Orchid Flower'],
        'Medium Orchid': ['Orchid Flower'],
        'Dark Turquoise': ['Turquoise Gem'],
        'Medium Turquoise': ['Turquoise Sea'],
        'Light Salmon': ['Salmon Roe'],
        'Dark Goldenrod': ['Goldenrod Flower'],
        'Medium Violet Red': ['Violet Petal'],
        'Light Green': ['Green Apple'],
        'Light Sky Blue': ['Sky'],
        'Light Steel Blue': ['Steel Metal'],
        'Pale Goldenrod': ['Goldenrod Flower'],
        'Light Yellow': ['Buttercup'],
        'Mint Cream': ['Mint Leaf'],
        'Honeydew': ['Honeydew Melon'],
        'Alice Blue': ['Bluebell'],
        'Ghost White': ['Ghost Orchid'],
        'Snow': ['Snowflake'],
        'Seashell': ['Seashell'],
        'Floral White': ['White Rose'],
        'Old Lace': ['Lace Dress'],
        'Antique White': ['Antique Vase'],
        'Papaya Whip': ['Papaya Fruit'],
        'Blanched Almond': ['Almond'],
        'Lavender Blush': ['Lavender Flower'],
        'Cornsilk': ['Corn Husk'],
        'Lemon Chiffon': ['Lemon Pie'],
        'Light Goldenrod Yellow': ['Goldenrod Flower'],
        'Light Cyan': ['Cyan Gem'],
        'Pale Green': ['Green Apple'],
        'Spring Green': ['Spring Leaves'],
        'Sea Green': ['Seaweed'],
        'Dark Green': ['Evergreen'],
        'Yellow Green': ['Green Banana'],
        'Olive Drab': ['Olive Tree'],
        'Dark Sea Green': ['Sea Turtle'],
        'Light Sea Green': ['Sea Glass'],
        'Light Goldenrod': ['Goldenrod Flower'],
        'Goldenrod': ['Goldenrod Flower'],
        'Rosy Brown': ['Rosy Cheeks'],
        'Burlywood': ['Burlap'],
        'Navajo White': ['Navajo Blanket'],
        'Peach Puff': ['Peach Fruit'],
        'Light Pink': ['Pink Rose'],
        'Deep Pink': ['Pink Hibiscus'],
        'Hot Pink': ['Pink Flamingo'],
        'Violet Red': ['Violet Flower'],
        'Medium Slate Blue': ['Slate Stone'],
        'Cornflower Blue': ['Cornflower'],
        'Royal Blue': ['Royal Ribbon'],
        'Dodger Blue': ['Dodger Stadium'],
        'Deep Sky Blue': ['Sky'],
        'Light Blue': ['Bluebell'],
        'Powder Blue': ['Powder Puff'],
        'Pale Turquoise': ['Turquoise Gem'],
        'Sky Blue': ['Sky'],
        'Pale Goldenrod': ['Goldenrod Flower'],
        'Peach Puff': ['Peach Fruit'],
        'Pale Green': ['Green Apple'],
        'Pale Turquoise': ['Turquoise Gem'],
        'Charcoal': ['Charcoal Briquette'],
        'Dark Gray': ['Gray Cloud'],
        'Light Gray': ['Gray Cloud'],
        'Ash Gray': ['Ash Tree'],
        'Jet': ['Jet Plane'],
        'Smoky Black': ['Smoky Quartz'],
        'Outer Space': ['Outer Space'],
        'Gun Metal': ['Gun Barrel'],
        'Timberwolf': ['Timberwolf Fur'],
        'Platinum': ['Platinum Ring'],
        'Old Silver': ['Silver Coin'],
        'Khaki Drab': ['Khaki Pants'],
        'Desert Sand': ['Desert Dune'],
        'Pale Silver': ['Silver Coin'],
        'Shadow': ['Shadow'],
        'Pewter Blue': ['Pewter Mug'],
        'Cool Black': ['Black Ice'],
        'Cerulean': ['Cerulean Sky'],
        'Ultramarine': ['Ultramarine Sea'],
        'Electric Blue': ['Blue Neon Light'],
        'Jade': ['Jade Stone'],
        'Persian Blue': ['Persian Rug'],
        'Cerulean Blue': ['Cerulean Sky'],
        'Egyptian Blue': ['Egyptian Painting'],
        'Klein Blue': ['Klein Bottle'],
        'Space Cadet': ['Cadet Uniform'],
        'Charcoal': ['Charcoal Briquette'],
        'Dark Gray': ['Gray Cloud'],
        'Light Gray': ['Gray Cloud'],
        'Outer Space': ['Outer Space'],
        'Old Silver': ['Coin'],
        'Khaki Drab': ['Dress Pants'],
        'Desert Sand': ['Desert Dune'],
        'Rosy Brown': ['Rosy Cheeks'],
        'Shadow': ['Shadow'],
        'Saddle Brown': ['Saddle Leather'],
        'Chocolate': ['Chocolate Bar'],
        'Peru': ['Rich Soil', 'Certain Types of Rock'],
        'Apple Green': ['Apple', 'Green Bell Pepper'],
        'Apricot': ['Apricot', 'Peach'],
        'Aqua': ['Swimming Pool', 'Tropical Sea'],
        'Army Green': ['Army Uniform', 'Camouflage Gear'],
        'Asparagus': ['Asparagus', 'Broccoli'],
        'Atomic Tangerine': ['Tangerine', 'Orange Soda'],
        'Baker-Miller Pink': ['Bubblegum', 'Cotton Candy'],
        'Banana Yellow': ['Banana', 'Yellow Balloon'],
        'Barn Red': ['Barn', 'Old Brick House'],
        'Battleship Grey': ['Battleship', 'Military Equipment'],
        'Bazaar': ['Vintage Sofa', 'Old Photograph'],
        'Blizzard Blue': ['Winter Jacket', 'Ice Sculpture'],
        'Blue Bell': ['Bluebell Flower', 'Lavender Plant'],
        'Blue Green': ['Peacock Feather', 'Turquoise Jewelry'],
        'Blue Violet': ['Violet Flower', 'Amethyst Crystal'],
        'Blush': ['Blush Makeup', 'Rose Petal'],
        'Bole': ['Tree Bark', 'Clay Pot'],
        'Brick Red': ['Brick Wall', 'Clay Roof Tile'],
        'Bright Green': ['Neon Sign', 'Highlighter'],
        'Bright Lilac': ['Lilac Flower', 'Easter Egg'],
        'Bright Maroon': ['Maroon Sweater', 'Plum'],
        'Bright Navy Blue': ['Sailor Uniform', 'Business Suit'],
        'Bright Pink': ['Flamingo', 'Pink Neon Sign'],
        'Bright Turquoise': ['Turquoise Necklace', 'Beach Towel'],
        'Bright Ube': ['Ube Halaya', 'Purple Yam'],
        'Bright Yellow': ['Sunflower', 'Yellow Raincoat'],
        'Brilliant Rose': ['Rose Flower', 'Lipstick'],
        'Brink Pink': ['Pink Crayon', 'Carnation'],
        'British Racing Green': ['Race Car', 'Hunter Boots'],
        'Bronze': ['Bronze Statue', 'Bronze Medal'],
        'Brown Sugar': ['Brown Sugar', 'Cinnamon Stick'],
        'Bud Green': ['Leaf Bud', 'Green Bean'],
        'Buff': ['Buff Paper', 'Beige Sweater'],
        'Burgundy': ['Wine', 'Burgundy Velvet Dress'],
        'Byzantine': ['Byzantine Mosaic', 'Purple Robe'],
        'Byzantium': ['Byzantium Tapestry', 'Royal Garment'],
        'Cambridge Blue': ['University Scarf', 'Vintage Book'],
        'Camel': ['Camel Coat', 'Sand Dune'],
        'Capri': ['Capri Pants', 'Ocean Wave'],
        'Cardinal': ['Cardinal Bird', 'Red Vestment'],
        'Caribbean Green': ['Tropical Sea', 'Green Parrot'],
        'Carmine': ['Carmine Ink', 'Cherry'],
        'Carnation Pink': ['Carnation Flower', 'Birthday Cake Frosting'],
        'Carnelian': ['Carnelian Gemstone', 'Pomegranate'],
        'Carrot Orange': ['Carrot', 'Orange Juice'],
        'Ceil': ['Ceiling Paint', 'Soft Sky'],
        'Celadon': ['Celadon Pottery', 'Pistachio'],
        'Celeste': ['Celeste Cloud', 'Sky Blue Dress'],
        'Cerise': ['Cerise Ribbon', 'Cherry Blossom'],
        'Cerulean Frost': ['Winter Sky', 'Frosted Window'],
        'Champagne': ['Champagne', 'Champagne Dress'],
        'Charcoal': ['Charcoal Drawing', 'BBQ Charcoal'],
        'Cherry': ['Cherry', 'Cherry Lip Gloss'],
        'Chestnut': ['Chestnut', 'Wooden Furniture'],
        'China Pink': ['China Plate', 'Cherry Blossom'],
        'Chinese Red': ['Chinese Lantern', 'Red Envelope'],
        'Chocolate': ['Chocolate Bar', 'Brownie'],
        'Cinereous': ['Old Photograph', 'Vintage Dress'],
        'Cinnabar': ['Cinnabar Gemstone', 'Red Ink'],
        'Cinnamon': ['Cinnamon Stick', 'Cinnamon Roll'],
        'Citrine': ['Citrine Crystal', 'Lemon'],
        'Classic Rose': ['Classic Rose Flower', 'Blush'],
        'Cobalt Blue': ['Cobalt Glass', 'Blue Ink'],
        'Columbia Blue': ['Columbia University T-Shirt', 'Light Blue Sweater'],
        'Congo Pink': ['Flamingo', 'Pink Bubblegum'],
        'Copper': ['Copper Wire', 'Penny'],
        'Copper Rose': ['Copper Rose', 'Metallic Paint'],
        'Coquelicot': ['Poppy Flower', 'Red Paint'],
        'Coral Pink': ['Coral Reef', 'Pink Sand'],
        'Cordovan': ['Horse Fur', 'Leather Wallet'],
        'Cornflower': ['Cornflower Flower', 'Blue Sky'],
        'Cosmic Cobalt': ['Galaxy Print', 'Night Sky'],
        'Cosmic Latte': ['Milky Way', 'Latte Foam'],
        'Cotton Candy': ['Cotton Candy', 'Pink Cloud'],
        'Cream': ['Cream', 'Vanilla Ice Cream'],
        'Crimson Red': ['Blood', 'Crimson Fabric'],
        'Cyan Blue': ['Tropical Water', 'Cyan Light'],
        'Cyber Grape': ['Grape Soda', 'Purple LED'],
        'Cyber Yellow': ['Yellow LED', 'Construction Vest'],
        'Daffodil': ['Daffodil Flower', 'Easter Decoration'],
        'Dandelion': ['Dandelion Flower', 'Yellow Balloon'],
        'Dark Brown': ['Chocolate Cake', 'Dark Wood'],
        'Dark Candy Apple Red': ['Candy Apple', 'Red Car'],
        'Dark Cerulean': ['Deep Ocean', 'Navy Uniform'],
        'Dark Chestnut': ['Horse', 'Chestnut Tree'],
        'Dark Coral': ['Coral Reef', 'Tropical Fish'],
        'Dark Cyan': ['Teal Sofa', 'Aquarium'],
        'Dark Electric Blue': ['Electric Guitar', 'Neon Sign'],
        'Dark Imperial Blue': ['Royal Robe', 'Evening Sky'],
        'Dark Jungle Green': ['Forest', 'Jungle Camouflage'],
        'Dark Lava': ['Volcanic Rock', 'Obsidian'],
        'Dark Lavender': ['Lavender Sachet', 'Purple Blanket'],
        'Dark Liver': ['Liver', 'Dark Brown Sofa'],
        'Dark Magenta': ['Magenta Flower', 'Purple Scarf'],
        'Dark Medium Gray': ['Slate Roof', 'Stone Path'],
        'Dark Midnight Blue': ['Night Sky', 'Dark Denim'],
        'Dark Moss Green': ['Moss', 'Forest Canopy'],
        'Dark Orchid': ['Orchid Flower', 'Purple Velvet'],
        'Dark Pastel Green': ['Mint Leaf', 'Green Smoothie'],
        'Dark Red': ['Red Wine', 'Ruby'],
        'Dark Scarlet': ['Scarlet Robe', 'Red Curtain'],
        'Dark Sienna': ['Old Brick', 'Clay Pot'],
        'Dark Spring Green': ['Spring Grass', 'Leaf'],
        'Dark Tan': ['Leather Belt', 'Brown Jacket'],
        'Dark Tangerine': ['Tangerine', 'Orange Peel'],
        'Dark Terracotta': ['Terracotta Pot', 'Roof Tile'],
        'Dark Violet': ['Violet Flower', 'Purple Velvet'],
        'Jet Black': ['Night', 'Black hair'],
        'Ebony': ['Bones'],
        'Onyx': ['Gem'],
        'Obsidian': ['Gem'],
        'Dark Lava': ['Lava'],
        'Charcoal': ['Charcoal Briquette'],
        'Dark Gray': ['Gray Cloud'],
        'Light Gray': ['Gray Cloud'],
        'Outer Space': ['Outer Space'],
        'Old Silver': ['Coin'],
        'Khaki Drab': ['Dress Pants'],
        'Desert Sand': ['Desert Dune'],
        'Rosy Brown': ['Rosy Cheeks'],
        'Shadow': ['Shadow'],
        'Saddle Brown': ['Saddle Leather'],
        'Chocolate': ['Chocolate Bar'],
        'Peru': ['Rich Soil', 'Certain Types of Rock'],
        'Apple Green': ['Apple', 'Green Bell Pepper'],
        'Apricot': ['Apricot', 'Peach'],
        'Eerie Black': ['Dark Forest'],
        'Gun metal': ['Gun'],
        'Davys Grey': ['Bass Fish'],
        'Raisin Black': ['Raisin']
    }

def get_color_makeup():
    return {
    'Red': ['Red-100%'],
    'Green': ['Green-100%'],
    'Blue': ['Blue-100%'],
    'Yellow': ['Yellow-100%'],
    'Cyan': ['Blue-50%', 'Green-50%'],
    'Magenta': ['Red-50%', 'Blue-50%'],
    'White': ['Red-33%', 'Green-33%', 'Blue-33%'],
    'Black': ['Black-100%'],
    'Gray': ['White-50%', 'Black-50%'],
    'Orange': ['Red-50%', 'Yellow-50%'],
    'Pink': ['Red-50%', 'White-50%'],
    'Purple': ['Red-50%', 'Blue-50%'],
    'Brown': ['Red-50%', 'Green-25%', 'Blue-25%'],
    'Maroon': ['Red-50%', 'Black-50%'],
    'Olive': ['Yellow-50%', 'Black-50%'],
    'Lime': ['Green-100%'],
    'Teal': ['Green-50%', 'Blue-50%'],
    'Navy': ['Blue-50%', 'Black-50%'],
    'Aquamarine': ['Green-50%', 'Blue-50%'],
    'Azure': ['Blue-50%', 'White-50%'],
    'Beige': ['Yellow-50%', 'White-50%'],
    'Bisque': ['Orange-50%', 'White-50%'],
    'Chocolate': ['Brown-100%'],
    'Coral': ['Red-50%', 'Orange-50%'],
    'Crimson': ['Red-60%', 'Blue-20%', 'Purple-20%'],
    'Fuchsia': ['Red-50%', 'Blue-50%'],
    'Gold': ['Yellow-70%', 'Orange-30%'],
    'Ivory': ['White-100%'],
    'Khaki': ['Yellow-70%', 'Green-30%'],
    'Lavender': ['Purple-70%', 'White-30%'],
    'Linen': ['White-80%', 'Yellow-20%'],
    'Moccasin': ['Yellow-50%', 'White-50%'],
    'Orchid': ['Purple-70%', 'Pink-30%'],
    'Peru': ['Orange-50%', 'Brown-30%', 'Red-20%'],
    'Plum': ['Purple-70%', 'Red-30%'],
    'Salmon': ['Red-60%', 'Orange-40%'],
    'Sienna': ['Brown-70%', 'Red-30%'],
    'Silver': ['Gray-100%'],
    'Tan': ['Brown-70%', 'Yellow-30%'],
    'Thistle': ['Purple-50%', 'White-50%'],
    'Tomato': ['Red-80%', 'Orange-20%'],
    'Turquoise': ['Blue-50%', 'Green-50%'],
    'Violet': ['Blue-50%', 'Red-50%'],
    'Wheat': ['Yellow-70%', 'White-30%'],
    'Sky Blue': ['Blue-70%', 'White-30%'],
    'Forest Green': ['Green-100%'],
    'Indian Red': ['Red-60%', 'Brown-40%'],
    'Steel Blue': ['Blue-70%', 'Gray-30%'],
    'Dark Slate Blue': ['Blue-70%', 'Gray-30%'],
    'Medium Spring Green': ['Green-80%', 'Yellow-20%'],
    'Rosy Brown': ['Brown-70%', 'Red-30%'],
    'Dark Olive Green': ['Green-70%', 'Yellow-30%'],
    'Saddle Brown': ['Brown-100%'],
    'Medium Purple': ['Purple-70%', 'Blue-30%'],
    'Medium Sea Green': ['Green-70%', 'Blue-30%'],
    'Fire Brick': ['Red-70%', 'Brown-30%'],
    'Cadet Blue': ['Blue-50%', 'Gray-50%'],
    'Pale Violet Red': ['Red-60%', 'Purple-40%'],
    'Dark Khaki': ['Yellow-50%', 'Green-30%', 'Red-20%'],
    'Light Coral': ['Red-70%', 'Orange-30%'],
    'Dark Salmon': ['Red-60%', 'Orange-40%'],
    'Medium Aquamarine': ['Blue-40%', 'Green-40%', 'Yellow-20%'],
    'Dark Orchid': ['Purple-70%', 'Red-30%'],
    'Medium Orchid': ['Purple-60%', 'Red-40%'],
    'Dark Turquoise': ['Blue-50%', 'Green-40%', 'Yellow-10%'],
    'Medium Turquoise': ['Blue-50%', 'Green-40%', 'Yellow-10%'],
    'Light Salmon': ['Red-50%', 'Orange-30%', 'Yellow-20%'],
    'Dark Goldenrod': ['Yellow-60%', 'Orange-30%', 'Red-10%'],
    'Medium Violet Red': ['Purple-50%', 'Red-50%'],
    'Light Green': ['Green-50%', 'Yellow-30%', 'Blue-20%'],
    'Light Sky Blue': ['Blue-70%', 'Green-20%', 'Yellow-10%'],
    'Light Steel Blue': ['Blue-60%', 'Green-20%', 'Yellow-20%'],
    'Pale Goldenrod': ['Yellow-50%', 'Green-30%', 'Red-20%'],
    'Light Yellow': ['Yellow-70%', 'Green-20%', 'Red-10%'],
    'Mint Cream': ['Green-50%', 'Yellow-30%', 'Blue-20%'],
    'Honeydew': ['Green-50%', 'Yellow-30%', 'Blue-20%'],
    'Alice Blue': ['Blue-70%', 'Green-20%', 'Yellow-10%'],
    'Ghost White': ['Blue-60%', 'Green-20%', 'Yellow-20%'],
    'Snow': ['White-100%'],
    'Seashell': ['White-80%', 'Pink-20%'],
    'Floral White': ['White-100%'],
    'Old Lace': ['White-100%'],
    'Antique White': ['White-80%', 'Yellow-20%'],
    'Papaya Whip': ['White-60%', 'Yellow-40%'],
    'Blanched Almond': ['White-50%', 'Yellow-30%', 'Red-20%'],
    'Lavender Blush': ['Purple-70%', 'White-30%'],
    'Cornsilk': ['Yellow-70%', 'White-30%'],
    'Lemon Chiffon': ['Yellow-70%', 'White-30%'],
    'Light Goldenrod Yellow': ['Yellow-70%', 'White-30%'],
    'Light Cyan': ['Cyan-100%'],
    'Pale Green': ['Green-70%', 'Yellow-20%', 'Blue-10%'],
    'Spring Green': ['Green-80%', 'Yellow-20%'],
    'Sea Green': ['Green-70%', 'Blue-20%', 'Yellow-10%'],
    'Dark Green': ['Green-100%'],
    'Yellow Green': ['Green-70%', 'Yellow-30%'],
    'Olive Drab': ['Green-50%', 'Yellow-40%', 'Red-10%'],
    'Dark Sea Green': ['Green-60%', 'Yellow-30%', 'Blue-10%'],
    'Light Sea Green': ['Green-50%', 'Blue-30%', 'Yellow-20%'],
    'Light Goldenrod': ['Yellow-60%', 'White-40%'],
    'Goldenrod': ['Yellow-60%', 'Orange-30%', 'Red-10%'],
    'Rosy Brown': ['Brown-60%', 'Red-30%', 'White-10%'],
    'Burlywood': ['Brown-60%', 'Yellow-30%', 'White-10%'],
    'Navajo White': ['White-70%', 'Yellow-20%', 'Red-10%'],
    'Peach Puff': ['White-50%', 'Yellow-30%', 'Red-20%'],
    'Light Pink': ['Pink-70%', 'White-30%'],
    'Deep Pink': ['Pink-80%', 'Red-20%'],
    'Hot Pink': ['Pink-80%', 'Red-20%'],
    'Violet Red': ['Red-60%', 'Purple-40%'],
    'Medium Slate Blue': ['Blue-60%', 'Purple-40%'],
    'Cornflower Blue': ['Blue-70%', 'Purple-30%'],
    'Royal Blue': ['Blue-80%', 'Purple-20%'],
    'Dodger Blue': ['Blue-80%', 'Green-20%'],
    'Deep Sky Blue': ['Blue-80%', 'Green-20%'],
    'Light Blue': ['Blue-70%', 'Green-30%'],
    'Powder Blue': ['Blue-70%', 'Green-30%'],
    'Pale Turquoise': ['Blue-70%', 'Green-30%'],
    'Sky Blue': ['Blue-70%', 'Green-30%'],
    'Pale Goldenrod': ['Yellow-70%', 'White-30%'],
    'Peach Puff': ['White-50%', 'Yellow-30%', 'Red-20%'],
    'Pale Green': ['Green-70%', 'Yellow-20%', 'Blue-10%'],
    'Pale Turquoise': ['Blue-70%', 'Green-30%'],
    'Charcoal': ['Black-100%'],
    'Dark Gray': ['Gray-100%'],
    'Light Gray': ['Gray-100%'],
    'Ash Gray': ['Gray-70%', 'White-30%'],
    'Jet': ['Black-100%'],
    'Smoky Black': ['Black-100%'],
    'Outer Space': ['Black-80%', 'White-20%'],
    'Gun Metal': ['Black-70%', 'White-30%'],
    'Timberwolf': ['Gray-70%', 'White-30%'],
    'Platinum': ['Gray-80%', 'White-20%'],
    'Old Silver': ['Gray-70%', 'White-30%'],
    'Khaki Drab': ['Green-50%', 'Yellow-30%', 'Red-20%'],
    'Desert Sand': ['Yellow-70%', 'Brown-30%'],
    'Pale Silver': ['Gray-70%', 'White-30%'],
    'Shadow': ['Gray-60%', 'Black-40%'],
    'Pewter Blue': ['Blue-50%', 'Gray-50%'],
    'Cool Black': ['Black-100%'],
    'Cerulean': ['Blue-70%', 'Green-30%'],
    'Ultramarine': ['Blue-80%', 'Purple-20%'],
    'Electric Blue': ['Blue-60%', 'Green-40%'],
    'Jade': ['Green-70%', 'Blue-30%'],
    'Persian Blue': ['Blue-70%', 'Purple-30%'],
    'Cerulean Blue': ['Blue-70%', 'Green-30%'],
    'Egyptian Blue': ['Blue-70%', 'Green-30%'],
    'Klein Blue': ['Blue-70%', 'Purple-30%'],
    'Space Cadet': ['Blue-60%', 'Purple-40%'],
    'Cinereous': ['Gray-70%', 'White-30%'],
    'Cinnabar': ['Red-70%', 'Orange-30%'],
    'Cinnamon': ['Brown-70%', 'Red-30%'],
    'Citrine': ['Yellow-80%', 'Green-20%'],
    'Classic Rose': ['Pink-70%', 'White-30%'],
    'Cobalt Blue': ['Blue-80%', 'Purple-20%'],
    'Columbia Blue': ['Blue-70%', 'Green-30%'],
    'Congo Pink': ['Pink-70%', 'White-30%'],
    'Copper': ['Brown-70%', 'Red-30%'],
    'Copper Rose': ['Brown-70%', 'Red-30%'],
    'Coquelicot': ['Red-70%', 'Orange-30%'],
    'Coral Pink': ['Pink-70%', 'White-30%'],
    'Cordovan': ['Brown-70%', 'Red-30%'],
    'Cornflower': ['Blue-70%', 'Purple-30%'],
    'Cosmic Cobalt': ['Blue-70%', 'Purple-30%'],
    'Cosmic Latte': ['White-100%'],
    'Cotton Candy': ['Pink-70%', 'White-30%'],
    'Cream': ['White-80%', 'Yellow-20%'],
    'Crimson Red': ['Red-80%', 'Purple-20%'],
    'Cyan Blue': ['Blue-60%', 'Green-40%'],
    'Cyber Grape': ['Purple-70%', 'Blue-30%'],
    'Cyber Yellow': ['Yellow-80%', 'Green-20%'],
    'Daffodil': ['Yellow-80%', 'Green-20%'],
    'Dandelion': ['Yellow-80%', 'Green-20%'],
    'Dark Brown': ['Brown-100%'],
    'Dark Candy Apple Red': ['Red-70%', 'Purple-30%'],
    'Dark Cerulean': ['Blue-70%', 'Green-30%'],
    'Dark Chestnut': ['Brown-70%', 'Red-30%'],
    'Dark Coral': ['Red-70%', 'Orange-30%'],
    'Dark Cyan': ['Green-70%', 'Blue-30%'],
    'Dark Electric Blue': ['Blue-70%', 'Green-30%'],
    'Dark Imperial Blue': ['Blue-70%', 'Purple-30%'],
    'Dark Jungle Green': ['Green-100%'],
    'Dark Lava': ['Brown-70%', 'Black-30%'],
    'Dark Lavender': ['Purple-70%', 'Blue-30%'],
    'Dark Liver': ['Brown-70%', 'Gray-30%'],
    'Dark Magenta': ['Purple-80%', 'Red-20%'],
    'Dark Medium Gray': ['Gray-100%'],
    'Dark Midnight Blue': ['Blue-70%', 'Black-30%'],
    'Dark Moss Green': ['Green-100%'],
    'Dark Pastel Green': ['Green-80%', 'Yellow-20%'],
    'Dark Red': ['Red-100%'],
    'Dark Scarlet': ['Red-80%', 'Purple-20%'],
    'Dark Sienna': ['Brown-100%'],
    'Dark Spring Green': ['Green-100%'],
    'Dark Tan': ['Brown-80%', 'Yellow-20%'],
    'Dark Tangerine': ['Orange-80%', 'Yellow-20%'],
    'Dark Terracotta': ['Brown-70%', 'Orange-30%'],
    'Dark Violet': ['Purple-100%'],
    'Jet Black': ['Black-100%'],
    'Ebony': ['Black-100%'],
    'Onyx': ['Black-100%'],
    'Obsidian': ['Black-100%'],
    'Dark Lava': ['Brown-70%', 'Black-30%'],
    'Apple Green': ['Green-80%', 'Yellow-20%'],
    'Apricot': ['Orange-70%', 'Yellow-30%'],
    'Aqua': ['Cyan-100%'],
    'Army Green': ['Green-80%', 'Black-20%'],
    'Asparagus': ['Green-70%', 'Yellow-30%'],
    'Atomic Tangerine': ['Orange-70%', 'Red-30%'],
    'Baker-Miller Pink': ['Pink-70%', 'Red-30%'],
    'Banana Yellow': ['Yellow-80%', 'Green-20%'],
    'Barn Red': ['Red-80%', 'Brown-20%'],
    'Battleship Grey': ['Gray-100%'],
    'Bazaar': ['Brown-70%', 'Red-30%'],
    'Blizzard Blue': ['Blue-70%', 'White-30%'],
    'Blue Bell': ['Blue-70%', 'Purple-30%'],
    'Blue Green': ['Blue-70%', 'Green-30%'],
    'Blue Violet': ['Blue-70%', 'Purple-30%'],
    'Blush': ['Pink-70%', 'Red-30%'],
    'Bole': ['Brown-70%', 'Red-30%'],
    'Brick Red': ['Red-70%', 'Brown-30%'],
    'Bright Green': ['Green-100%'],
    'Bright Lilac': ['Purple-70%', 'Pink-30%'],
    'Bright Maroon': ['Red-70%', 'Purple-30%'],
    'Bright Navy Blue': ['Blue-70%', 'Black-30%'],
    'Bright Pink': ['Pink-100%'],
    'Bright Turquoise': ['Blue-70%', 'Green-30%'],
    'Bright Ube': ['Purple-70%', 'Pink-30%'],
    'Bright Yellow': ['Yellow-100%'],
    'Brilliant Rose': ['Pink-70%', 'Purple-30%'],
    'Brink Pink': ['Pink-70%', 'Red-30%'],
    'British Racing Green': ['Green-80%', 'Black-20%'],
    'Bronze': ['Brown-70%', 'Yellow-30%'],
    'Brown Sugar': ['Brown-70%', 'Yellow-30%'],
    'Bud Green': ['Green-80%', 'Yellow-20%'],
    'Buff': ['Yellow-70%', 'White-30%'],
    'Burgundy': ['Red-70%', 'Purple-30%'],
    'Byzantine': ['Purple-70%', 'Pink-30%'],
    'Byzantium': ['Purple-100%'],
    'Cambridge Blue': ['Blue-70%', 'Green-30%'],
    'Camel': ['Brown-70%', 'Yellow-30%'],
    'Capri': ['Blue-70%', 'Green-30%'],
    'Cardinal': ['Red-80%', 'Brown-20%'],
    'Caribbean Green': ['Green-80%', 'Blue-20%'],
    'Carmine': ['Red-80%', 'Purple-20%'],
    'Carnation Pink': ['Pink-70%', 'White-30%'],
    'Carnelian': ['Red-80%', 'Brown-20%'],
    'Carrot Orange': ['Orange-70%', 'Yellow-30%'],
    'Ceil': ['Blue-70%', 'White-30%'],
    'Celadon': ['Green-70%', 'White-30%'],
    'Celeste': ['Blue-70%', 'White-30%'],
    'Cerise': ['Pink-70%', 'Red-30%'],
    'Cerulean Frost': ['Blue-70%', 'Gray-30%'],
    'Champagne': ['Yellow-70%', 'White-30%'],
    'Cherry': ['Red-80%', 'Pink-20%'],
    'Chestnut': ['Brown-80%', 'Red-20%'],
    'China Pink': ['Pink-70%', 'Red-30%'],
    'Chinese Red': ['Red-80%', 'Brown-20%'],
    'Chocolate (Traditional)': ['Brown-100%'],
    'Cinereous': ['Gray-70%', 'White-30%'],
    'Cinnabar': ['Red-70%', 'Orange-30%'],
    'Cinnamon': ['Brown-70%', 'Red-30%'],
    'Citrine': ['Yellow-80%', 'Green-20%'],
    'Classic Rose': ['Pink-70%', 'White-30%'],
    'Cobalt Blue': ['Blue-80%', 'Purple-20%'],
    'Columbia Blue': ['Blue-70%', 'Green-30%'],
    'Congo Pink': ['Pink-70%', 'White-30%'],
    'Copper': ['Brown-70%', 'Red-30%'],
    'Copper Rose': ['Brown-70%', 'Red-30%'],
    'Coquelicot': ['Red-70%', 'Orange-30%'],
    'Coral Pink': ['Pink-70%', 'White-30%'],
    'Cordovan': ['Brown-70%', 'Red-30%'],
    'Cornflower': ['Blue-70%', 'Purple-30%'],
    'Cosmic Cobalt': ['Blue-70%', 'Purple-30%'],
    'Cosmic Latte': ['White-100%'],
    'Cotton Candy': ['Pink-70%', 'White-30%'],
    'Cream': ['White-80%', 'Yellow-20%'],
    'Crimson Red': ['Red-80%', 'Purple-20%'],
    'Cyan Blue': ['Blue-60%', 'Green-40%'],
    'Cyber Grape': ['Purple-70%', 'Blue-30%'],
    'Cyber Yellow': ['Yellow-80%', 'Green-20%'],
    'Daffodil': ['Yellow-80%', 'Green-20%'],
    'Dandelion': ['Yellow-80%', 'Green-20%'],
    'Dark Brown': ['Brown-100%'],
    'Dark Candy Apple Red': ['Red-70%', 'Purple-30%'],
    'Dark Cerulean': ['Blue-70%', 'Green-30%'],
    'Dark Chestnut': ['Brown-70%', 'Red-30%'],
    'Dark Coral': ['Red-70%', 'Orange-30%'],
    'Dark Cyan': ['Green-70%', 'Blue-30%'],
    'Dark Electric Blue': ['Blue-70%', 'Green-30%'],
    'Dark Imperial Blue': ['Blue-70%', 'Purple-30%'],
    'Dark Jungle Green': ['Green-100%'],
    'Dark Lava': ['Brown-70%', 'Black-30%'],
    'Dark Lavender': ['Purple-70%', 'Blue-30%'],
    'Dark Liver': ['Brown-70%', 'Gray-30%'],
    'Dark Magenta': ['Purple-80%', 'Red-20%'],
    'Dark Medium Gray': ['Gray-100%'],
    'Dark Midnight Blue': ['Blue-70%', 'Black-30%'],
    'Dark Moss Green': ['Green-100%'],
    'Dark Pastel Green': ['Green-80%', 'Yellow-20%'],
    'Dark Red': ['Red-100%'],
    'Dark Scarlet': ['Red-80%', 'Purple-20%'],
    'Dark Sienna': ['Brown-100%'],
    'Dark Spring Green': ['Green-100%'],
    'Dark Tan': ['Brown-80%', 'Yellow-20%'],
    'Dark Tangerine': ['Orange-80%', 'Yellow-20%'],
    'Dark Terracotta': ['Brown-70%', 'Orange-30%'],
    'Dark Violet': ['Purple-100%'],
    'Gunmetal': ['Green-70%', 'Black-30%'],
    'Eerie Black': ['Black-100%'],
    'Davys Grey': ['Green-30%', 'Grey-70%'],
    'Raisin Black': ['Black-100%']
 }