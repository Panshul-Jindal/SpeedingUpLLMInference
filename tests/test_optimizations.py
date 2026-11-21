from harness import optimizations as opt


def test_bits_roundtrip():
    for m in range(128):
        assert opt.from_bits(opt.to_bits(m)) == m
    assert opt.to_bits(0) == "0000000"
    assert opt.to_bits(opt.bit(1)) == "1000000"
    assert opt.to_bits(opt.bit(7)) == "0000001"


def test_names_roundtrip():
    for m in range(128):
        assert opt.from_names(opt.names(m)) == m
    assert opt.names(0) == "default"
    assert opt.names(opt.bit(1) | opt.bit(4)) == "quant+prefix"


def test_numeric_mask():
    assert opt.NUMERIC_MASK == 0b0000111
    assert [o.number for o in opt.OPTIMIZATIONS if o.numeric] == [1, 2, 3]
    assert opt.numeric_label(opt.from_bits("1011111")) == "101"
    assert opt.numeric_setting(opt.from_bits("0001111")) == 0


def test_active_mask():
    assert opt.active_numbers(1) == [1, 2, 3, 4, 5, 6]
    assert opt.active_numbers(4) == [1, 2, 3, 4, 5, 6, 7]
