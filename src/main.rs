use nannou::prelude::*;
use rand::prelude::*;
use std::num::Wrapping;
use std::fs::File;
use std::io::prelude::*;
use std::io::Cursor;
use byteorder::WriteBytesExt; // This trait adds methods to writeable types
use byteorder::ReadBytesExt;
use byteorder::BigEndian;
use std::env;
use std::time::{SystemTime};

fn main()
{
    nannou::app(model).update(update).run();
}

const ITERATIONS: u32 = 3000;

const CREATURES_PER_SIM: usize = 10;

const BOUNDS_X: f32 = 1000.0;
const BOUNDS_Y: f32 = 1000.0;
const RADIUS: f32 = 5.0;

const NUM_FOODS: i32 = 200;
const FOOD_VALUE: f32 = 30.0;

const NO_FOOD_SELECTED: usize = 9999999;

const PPM: f32 = 4.0;

const SUCCESS_RADIUS: f32 = 10.0;

fn is_alive(creature: &Creature) -> bool
{
    creature.energy > 0.0
    // colliding(&creature.point, &success_point, RADIUS, SUCCESS_RADIUS)
}

fn colliding(point1: &Point2, point2: &Point2, radius1: f32, radius2: f32) -> bool
{
    let dist_x = point1.x - point2.x;
    let dist_y = point1.y - point2.y;
    let distance = (dist_x*dist_x) + (dist_y*dist_y);

    let dist_r = radius1 + radius2;

    distance <= dist_r * dist_r
}

#[derive(Clone)]
struct Node
{
    weights: Vec<i16>,
    bias: f32
}

#[derive(Clone)]
struct Genetics
{
    inners: Vec<Node>,
    outputs: Vec<Node>
}

#[derive(Clone)]
struct Creature
{
    point: Point2,
    genetics: Genetics,
    energy: f32,
    selected_food: usize
}

impl Creature
{
    pub fn new(point: Point2, genetics: Genetics) -> Self
    {
        Creature { point, genetics, energy: 40.0, selected_food: NO_FOOD_SELECTED }
    }

    pub fn color(&self) -> Srgb<u8>
    {
        let mut r: f32 = 0.0;
        let mut g: f32 = 0.0;
        let mut b: f32 = 0.0;

        for node in self.genetics.inners.iter()
        {
            for i in 0..node.weights.len()
            {
                let weight = node.calc_weights(i);
                r += weight;
                g += weight + 0.5;
                b += weight + 0.8;
            }
            
            r += node.bias;
            g += node.bias + 0.5;
            b += node.bias + 0.8;
        }

        for node in self.genetics.outputs.iter()
        {
            for i in 0..node.weights.len()
            {
                let weight = node.calc_weights(i);
                r += weight;
                g += weight + 0.5;
                b += weight + 0.8;
            }
            
            r += node.bias;
            g += node.bias + 0.5;
            b += node.bias + 0.8;
        }

        r -= (r as i32) as f32;
        g -= (g as i32) as f32;
        b -= (b as i32) as f32;

        r *= 256.0;
        g *= 256.0;
        b *= 256.0;

        r += 128.0;
        g += 128.0;
        b += 128.0;

        if r < 40.0
        {
            r = 40.0;
        }
        if g < 40.0
        {
            g = 40.0;
        }
        if b < 40.0
        {
            b = 40.0;
        }

        Srgb { red: r as u8, green: g as u8, blue: b as u8, standard: ::core::marker::PhantomData }
    }
}

#[derive(Clone, Copy)]
struct Food
{
    value: f32,
    point: Point2
}

struct Model
{
    _window: window::Id,
    creatures: Vec<Creature>,
    rand: ThreadRng,
    itr: u32,
    generation: u32,
    success_point: Point2,
    foods: Vec<Option<Food>>,
    render: bool,
    time: SystemTime
}

impl Node
{
    pub fn calc_weights(self: &Node, i: usize) -> f32
    {
        // -4.0 to 4.0
        (self.weights[i] as f32) / ((i16::MAX / 4) as f32)
    }
}

fn save(genetics: &Genetics, name: String)
{
    let mut file = File::create(name).unwrap();
    file.write_u32::<BigEndian>(genetics.inners.len() as u32).unwrap();
    file.write_u32::<BigEndian>(genetics.outputs.len() as u32).unwrap();

    for node in genetics.inners.iter()
    {
        file.write_f32::<BigEndian>(node.bias).unwrap();
        file.write_u32::<BigEndian>(node.weights.len() as u32).unwrap();
        for f in node.weights.iter()
        {
            file.write_i16::<BigEndian>(*f).unwrap();
        }
    }

    for node in genetics.outputs.iter()
    {
        file.write_f32::<BigEndian>(node.bias).unwrap();
        file.write_u32::<BigEndian>(node.weights.len() as u32).unwrap();
        for f in node.weights.iter()
        {
            file.write_i16::<BigEndian>(*f).unwrap();
        }
    }

    file.flush().unwrap();
}

fn save_creatures(creatures: &Vec<&Creature>, save_id: String)
{
    std::fs::create_dir_all(format!("saves/{}/", save_id)).unwrap();

    for i in 0..creatures.len()
    {
        save(&creatures[i].genetics, format!("saves/{}/{}.bin", save_id, i));
    }
}

fn read(name: String) -> Genetics
{
    let mut file = File::open(name).unwrap();
    let mut v: Vec<u8> = Vec::new();
    
    file.read_to_end(&mut v).unwrap();

    let mut cursor = Cursor::new(v);
    
    let in_size = cursor.read_u32::<BigEndian>().unwrap();
    let out_size = cursor.read_u32::<BigEndian>().unwrap();

    let mut in_nodes: Vec<Node> = Vec::new();

    for _ in 0..in_size
    {
        let bias = cursor.read_f32::<BigEndian>().unwrap();
        let weights_len = cursor.read_u32::<BigEndian>().unwrap();

        let mut weights: Vec<i16> = Vec::new();

        for __ in 0..weights_len
        {
            weights.push(cursor.read_i16::<BigEndian>().unwrap());
        }

        in_nodes.push(Node { bias, weights });
    }

    let mut out_nodes: Vec<Node> = Vec::new();

    for _ in 0..out_size
    {
        let bias = cursor.read_f32::<BigEndian>().unwrap();
        let weights_len = cursor.read_u32::<BigEndian>().unwrap();

        let mut weights: Vec<i16> = Vec::new();

        for __ in 0..weights_len
        {
            weights.push(cursor.read_i16::<BigEndian>().unwrap());
        }

        out_nodes.push(Node { bias, weights });
    }

    Genetics {
        inners: in_nodes,
        outputs: out_nodes
    }
}

fn calc_weights(size: usize, rand: &mut ThreadRng) -> Vec<i16>
{
    let mut ret: Vec<i16> = Vec::new();
    for _ in 0..size
    {
        ret.push(rand.gen::<i16>());
    }
    ret
}

fn calc_nodes(size: usize, weights_size: usize, rand: &mut ThreadRng) -> Vec<Node>
{
    let mut ret = Vec::new();

    for _ in 0..size
    {
        ret.push(Node { weights: calc_weights(weights_size, rand), bias: rand.gen::<f32>() * 10.0 - 5.0 });
    }

    ret
}

fn init_creatures(rand: &mut ThreadRng) -> Vec<Creature>
{
    let args: Vec<String> = env::args().collect();

    if args.len() <= 1
    {
        println!("No arguments given - loading fresh generation!");

        let mut creatures: Vec<Creature> = Vec::new();
        for i in 0..CREATURES_PER_SIM
        {
            let theta = i as f32 / CREATURES_PER_SIM as f32 * 2.0 * PI;
            let s = theta.sin();
            let c = theta.cos();
            let radius = 250.0;

            creatures.push(Creature::new(pt2(
                radius * c, 
                radius * s), 
            Genetics
            {
                inners: calc_nodes(5, 3, rand), // 5 nodes getting data from 3 inputs
                outputs: calc_nodes(3, 5, rand) // 3 nodes getting data from 5 inputs
            }));
        }

        return creatures;
    }
    else
    {
        let gen = &args[1];
        println!("Loading Generation {}", gen);

        let err_msg = format!("Unable to find generation {}", gen);
        let size: usize = std::fs::read_dir(format!("saves/gen_{}/", gen))
                .expect(&err_msg).count();

        let mut creatures: Vec<Creature> = Vec::new();
        for i in 0..CREATURES_PER_SIM
        {
            let theta = i as f32 / CREATURES_PER_SIM as f32 * 2.0 * PI;
            let s = theta.sin();
            let c = theta.cos();
            let radius = 250.0;

            creatures.push(Creature::new(pt2(
                radius * c, 
                radius * s), 
                read(
                    format!("saves/gen_{}/{}.bin", gen, (rand.gen::<f32>() * size as f32) as usize)
                )
            ));
        }

        creatures
    }
}

fn model(app: &App) -> Model
{
    let window = app.new_window().view(view).size(1000, 1000).build().unwrap();
    let mut rand = rand::thread_rng();

    let args: Vec<String> = env::args().collect();
    let mut gen: u32 = 0;
    if args.len() == 2
    {
        gen = args[1].parse::<u32>().unwrap();
    }

    let mut m = Model { _window: window, creatures: init_creatures(&mut rand), rand, 
        itr: 0, generation: gen, success_point: pt2(0.0, 0.0), foods: Vec::new(), render: true,
        time: SystemTime::now() };

    m.foods = fill_foods(&mut m);

    m
}

// https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0
fn tanh(input: f32) -> f32
{
    (2.0 / (1.0 + f64::exp(-2.0 * (input as f64))) - 1.0) as f32
}

fn process_network(creature: &Creature, inputs: &Vec<f32>) -> Vec<f32>
{
    let mut outputs_inner = vec![0.0; creature.genetics.inners.len()];

    for i in 0..creature.genetics.inners.len()
    {
        for j in 0..inputs.len()
        {
            outputs_inner[i] += inputs[j] * creature.genetics.inners[i].calc_weights(j);
        }

        outputs_inner[i] = tanh(outputs_inner[i] + creature.genetics.inners[i].bias);
    }

    let mut outputs = vec![0.0; creature.genetics.outputs.len()];
    for i in 0..creature.genetics.outputs.len()
    {
        for j in 0..outputs_inner.len()
        {
            outputs[i] += outputs_inner[j] * creature.genetics.outputs[i].calc_weights(j);
        }

        outputs[i] = tanh(outputs[i] + creature.genetics.outputs[i].bias);
    }

    outputs
}

fn act_on_network(creature: &mut Creature, outputs: &Vec<f32>, rand: &mut ThreadRng)
{
    assert_eq!(outputs.len(), 3);

    let og_x = creature.point.x;
    let og_y = creature.point.y;

    creature.point.x += outputs[0] + outputs[2] * (2.0 * rand.gen::<f32>() - 1.0);
    creature.point.y += outputs[1] + outputs[2] * (2.0 * rand.gen::<f32>() - 1.0);

    creature.point.x = clamp(creature.point.x, -BOUNDS_X / 2.0 + RADIUS, BOUNDS_X / 2.0 - RADIUS);
    creature.point.y = clamp(creature.point.y, -BOUNDS_Y / 2.0 + RADIUS, BOUNDS_Y / 2.0 - RADIUS);

    let dx = (creature.point.x - og_x) / PPM;
    let dy = (creature.point.y - og_y) / PPM;

    let d = dx * dx + dy * dy;
    creature.energy -= 0.5 * d + 0.1;
}

fn mix_nodes(nodes_a: &Vec<Node>, nodes_b: &Vec<Node>, energy_a: f32, energy_b: f32, rand: &mut ThreadRng) -> Vec<Node>
{
    let mut nodes: Vec<Node> = Vec::new();

    for i in 0..nodes_a.len()
    {
        let mut bias: f32;
        if rand.gen::<f32>() * (energy_a + energy_b) < energy_a
        {
            bias = nodes_a[i].bias;
        }
        else
        {
            bias = nodes_b[i].bias;
        }

        if rand.gen::<f32>() < 0.05
        {
            bias += rand.gen::<f32>() * 2.0 - 1.0;
        }

        let mut weights: Vec<i16> = Vec::new();

        for j in 0..nodes_a[i].weights.len()
        {
            let mut weight: i16;
            if rand.gen::<f32>() * (energy_a + energy_b) < energy_a
            {
                weight = nodes_a[i].weights[j];
            }
            else
            {
                weight = nodes_b[i].weights[j];
            }

            if rand.gen::<f32>() < 0.05
            {
                weight = (Wrapping(weight) + Wrapping(rand.gen::<i16>() / 4)).0;
            }

            weights.push(weight);
        }

        nodes.push(Node
        {
            weights: weights,
            bias: bias
        });
    }

    nodes
}

fn mix_genes(gene_a: &Genetics, gene_b: &Genetics, energy_a: f32, energy_b: f32, rand: &mut ThreadRng) -> Genetics
{
    let inners: Vec<Node> = mix_nodes(&gene_a.inners, &gene_b.inners, energy_a, energy_b, rand);
    let outputs: Vec<Node> = mix_nodes(&gene_a.outputs, &gene_b.outputs, energy_a, energy_b, rand);

    Genetics
    {
        inners: inners,
        outputs: outputs 
    }
}

fn create_next_generation(model: &mut Model) -> Vec<Creature>
{
    let mut survivors: Vec<&Creature> = Vec::new();
    let mut creatures: Vec<Creature>;

    for creature in model.creatures.iter()
    {
        if is_alive(&creature)
        {
            survivors.push(creature);
        }
    }

    println!("Survivors: {}", survivors.len());

    println!("Saving Survivors...");

    save_creatures(&survivors, format!("gen_{}", model.generation));

    println!("Survivors Saved!");

    if survivors.len() < 2
    {
        creatures = init_creatures(&mut model.rand); // not enough to breed, try again
    }
    else
    {
        creatures = Vec::new();

        for i in 0..CREATURES_PER_SIM
        {
            let index_a: usize = (model.rand.gen::<f32>() * (survivors.len() as f32)) as usize;
            let mut index_b: usize = (model.rand.gen::<f32>() * (survivors.len() as f32)) as usize;
            while index_b == index_a
            {
                index_b = (model.rand.gen::<f32>() * (survivors.len() as f32)) as usize;
            }

            let parent_a = survivors[index_a];
            let parent_b = survivors[index_b];

            let genetics = mix_genes(&parent_a.genetics, &parent_b.genetics, 
                parent_a.energy, parent_b.energy, &mut model.rand);
            
            let theta = i as f32 / CREATURES_PER_SIM as f32 * 2.0 * PI;
            let s = theta.sin();
            let c = theta.cos();
            let radius = 250.0;
    
            let baby = Creature::new(
                pt2(
                    model.rand.gen::<f32>() * BOUNDS_X - BOUNDS_X / 2.0,//radius * c, 
                    model.rand.gen::<f32>() * BOUNDS_Y - BOUNDS_Y / 2.0,),
                genetics);

            creatures.push(baby);
        }
    }

    creatures
}

fn fill_foods(model: &mut Model) -> Vec<Option<Food>>
{
    let mut foods: Vec<Option<Food>> = Vec::new();
    
    for _ in 0..NUM_FOODS
    {
        let f = Food { value: FOOD_VALUE, point: 
            pt2(
                0.9 * (BOUNDS_X * model.rand.gen::<f32>() - (BOUNDS_X / 2.0)),
                0.9 * (BOUNDS_Y * model.rand.gen::<f32>() - (BOUNDS_Y / 2.0))
            ) };

        // let f = Food { value: FOOD_VALUE, point: 
        //     pt2(
        //         0.0,
        //         0.0
        //     ) };

        foods.push(Some(f));
    }

    foods
}

fn find_closest_food(foods: &Vec<Option<Food>>, point: &Point2) -> usize
{
    let mut best: usize = NO_FOOD_SELECTED;
    let mut best_dist = 0.0;
    
    for i in 0..foods.len()
    {
        if foods[i].is_none()
        {
            continue;
        }

        let food = foods[i].unwrap();

        let xx = food.point[0] - point[0];
        let yy = food.point[1] - point[1];
        let dist = xx * xx + yy * yy;
        if best == NO_FOOD_SELECTED
        {
            best_dist = dist;
            best = i;
        }
        else if dist < best_dist
        {
            best_dist = dist;
            best = i;
        }
    }

    best
}

fn update(_app: &App, model: &mut Model, _update: Update) 
{
    for key in _app.keys.down.iter()
    {
        if *key == Key::F11 && model.time.elapsed().unwrap().as_millis() > 250
        {
            model.render = !model.render;
            model.time = SystemTime::now();
        }
    }

    for creature in model.creatures.iter_mut()
    {
        if !is_alive(creature)
        {
            continue;
        }

        if creature.selected_food == NO_FOOD_SELECTED || 
            model.foods[creature.selected_food].is_none()
        {
            let food = find_closest_food(&model.foods, &creature.point);

            creature.selected_food = food;
        }

        {
            let i1: f32;
            let i2: f32;
            let i3: f32;

            if creature.selected_food == NO_FOOD_SELECTED || model.foods[creature.selected_food].is_none()
            {
                i1 = 0.0;
                i2 = 0.0;
                i3 = 0.0;
            }
            else
            {
                let food = model.foods[creature.selected_food].unwrap();

                i1 = food.point.x - creature.point.x; 
                i2 = food.point.y - creature.point.y; 
                i3 = 1.0;
            }

            act_on_network(creature, 
                &process_network(creature, &vec![i1, i2, i3]), 
                &mut model.rand);
        }

        {
            // for i in 0..model.foods.len()
            {
                let i = creature.selected_food;
                if i != NO_FOOD_SELECTED
                {
                    let food = model.foods[i];
                    if !food.is_none() 
                    {
                        let unwrapped = food.unwrap();

                        if colliding(&unwrapped.point, &creature.point, SUCCESS_RADIUS, RADIUS)
                        {
                            creature.energy += unwrapped.value;
                            model.foods[i] = None;
                        }
                    }
                }
            }
        }
    }

    if model.itr % 100 == 0
    {
        println!("Iteration {}", model.itr);
    }

    model.itr += 1;

    if model.itr >= ITERATIONS
    {
        model.creatures = create_next_generation(model);
        model.foods = fill_foods(model);
        model.itr = 0;
        model.generation += 1;

        model.success_point[0] = 0.9 * (BOUNDS_X * model.rand.gen::<f32>() - (BOUNDS_X / 2.0));
        model.success_point[1] = 0.9 * (BOUNDS_Y * model.rand.gen::<f32>() - (BOUNDS_Y / 2.0));

        println!("Generation {}", model.generation);
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let ref mut draw = app.draw();
    
    draw.background().color(BLACK);

    if model.render
    {
        // draw.ellipse().w_h(SUCCESS_RADIUS * 2.0, SUCCESS_RADIUS * 2.0).xy(model.success_point).color(PLUM);

        // let mut i = 0;
        for food in model.foods.iter()
        {
            if !food.is_none()
            {
                // if i == model.creatures[0].selected_food
                // {
                //     draw.ellipse().w_h(SUCCESS_RADIUS * 2.0, SUCCESS_RADIUS * 2.0)
                //         .xy(food.unwrap().point).color(BLUE);
                // }
                // else
                // {
                    draw.ellipse().w_h(SUCCESS_RADIUS * 2.0, SUCCESS_RADIUS * 2.0)
                        .xy(food.unwrap().point).color(RED);
                // }
            }
            // i += 1;
        }

        for creature in model.creatures.iter()
        {
            draw.ellipse().w_h(RADIUS * 2.0, RADIUS * 2.0)
                .xy(creature.point).color(creature.color());
        }
    }
    
    draw.to_frame(app, &frame).unwrap();
}