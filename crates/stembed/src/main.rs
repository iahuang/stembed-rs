use stembed::stemb::potion::PotionModel;
use stembed::tokenizer::Vocab;

pub fn main() {
    let model = PotionModel::potion_8m(
        Vocab::from_newline_deliminated_string(include_str!("../../../data/vocab.txt")),
        include_bytes!("../../../data/potion_base_8M.bin"),
        stembed::stemb::embedding::Endianness::Little,
    );
    let embeddings = model.embed_f32("Hello, world!");
    println!("{:?}", embeddings);
}
