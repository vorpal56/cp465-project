import { Component } from '@angular/core';

@Component({
  selector: 'article-search-engine',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
	tfidfResults: any[];
	secondaryIndexResults: any[];
	setSearchResults(allSearchResults: any) {
		this.tfidfResults = allSearchResults["tfidf"];
		this.secondaryIndexResults = allSearchResults["secondary_index"];
	}
}
