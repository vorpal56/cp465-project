import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'search-results',
  templateUrl: './results.component.html',
  styleUrls: ['./results.component.scss']
})
export class ResultsComponent implements OnInit {
	@Input("results") searchResults: any[];

  constructor() { }

  ngOnInit(): void {
  }
	print() {
		console.log(this.searchResults)
	}

}
